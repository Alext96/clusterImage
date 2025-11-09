import argparse, io, zipfile, os, shutil, math, warnings, csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import imagehash
import cv2
from tqdm import tqdm

# ML / clustering
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

# InsightFace (GPU-capable with onnxruntime-gpu)
import insightface
from insightface.app import FaceAnalysis

# ---------------------------
# Helpers / IO
# ---------------------------
IMG_EXT = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT

def read_images_from_zip(zip_path: Path, workdir: Path) -> List[Path]:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # preserve order as in zip listing
        members = [m for m in zf.namelist() if is_image(Path(m))]
        zf.extractall(workdir)
    return [workdir / m for m in members if is_image(Path(m))]

def collect_images(input_path: Path, tmpdir: Path) -> List[Path]:
    if input_path.suffix.lower() == ".zip":
        imgs = read_images_from_zip(input_path, tmpdir)
    else:
        imgs = [p for p in sorted(input_path.rglob("*")) if is_image(p)]
    if not imgs:
        raise SystemExit("No images found.")
    return imgs

# ---------------------------
# Visual feature extraction (scene/content)
# ---------------------------
class GlobalEmbedder(nn.Module):
    """ResNet50 backbone -> global 2048-d embedding (L2-normalized)."""
    def __init__(self, device="cpu"):
        super().__init__()
        try:
            weights = models.ResNet50_Weights.DEFAULT
            m = models.resnet50(weights=weights)
            try:
                _mean = list(weights.transforms().normalize.mean)
                _std  = list(weights.transforms().normalize.std)
            except Exception:
                _mean = [0.485, 0.456, 0.406]
                _std  = [0.229, 0.224, 0.225]
        except Exception:
            m = models.resnet50(pretrained=True)
            _mean = [0.485, 0.456, 0.406]
            _std  = [0.229, 0.224, 0.225]

        self.backbone = nn.Sequential(*list(m.children())[:-1])  # up to avgpool
        self.device = device
        self.eval().to(device)

        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mean, std=_std),
        ])

    @torch.inference_mode()
    def forward(self, pil: Image.Image) -> np.ndarray:
        x = self.tf(pil.convert("RGB")).unsqueeze(0).to(self.device)
        feat = self.backbone(x).squeeze().detach().cpu().numpy()  # 2048
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat

def color_hist_feature(pil: Image.Image, bins=32) -> np.ndarray:
    arr = np.asarray(pil.convert("RGB").resize((256,256)))
    hist = []
    for ch in range(3):
        h = cv2.calcHist([arr],[ch],None,[bins],[0,256]).flatten()
        h = h / (h.sum() + 1e-8)
        hist.append(h)
    return np.concatenate(hist)  # 96-d

def phash_feature(pil: Image.Image) -> np.ndarray:
    h = imagehash.phash(pil, hash_size=16)  # 16x16 -> 256 bits
    return np.array([int(b) for b in bin(int(str(h), 16))[2:].zfill(256)], dtype=np.float32)

def sharpness_score(pil: Image.Image) -> float:
    arr = np.asarray(pil.convert("L"))
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())

def exposure_score(pil: Image.Image) -> float:
    arr = np.asarray(pil.convert("L")) / 255.0
    mu, sigma = arr.mean(), arr.std()
    return float(np.exp(-((mu-0.5)**2)/0.02) * np.exp(-((sigma-0.25)**2)/0.02))

# ---------------------------
# InsightFace: face embeddings (GPU/CPU)
# ---------------------------
_FACE_APP = None

def _get_face_app(use_gpu: bool) -> FaceAnalysis:
    """Initialize InsightFace once. Prefer CUDA if requested and available."""
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    _FACE_APP = FaceAnalysis(name="buffalo_l", providers=providers)
    _FACE_APP.prepare(ctx_id=0)
    return _FACE_APP

def face_embedding(pil: Image.Image, use_gpu: bool) -> Optional[np.ndarray]:
    try:
        app = _get_face_app(use_gpu=use_gpu)
        arr = np.asarray(pil.convert("RGB"))
        faces = app.get(arr)
        if not faces:
            return None
        emb = faces[0].embedding.astype(np.float32)
        emb /= (np.linalg.norm(emb)+1e-8)
        return emb
    except Exception:
        return None

def face_box_area_fraction(pil: Image.Image, use_gpu: bool) -> float:
    try:
        app = _get_face_app(use_gpu=use_gpu)
        arr = np.asarray(pil.convert("RGB"))
        faces = app.get(arr)
        if not faces:
            return 0.0
        h, w = arr.shape[:2]
        areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
        return float(max(areas) / (h*w))
    except Exception:
        return 0.0

# ---------------------------
# Clustering
# ---------------------------
def auto_kmeans(X: np.ndarray, k_min=2, k_max=8, seed=42) -> Tuple[np.ndarray, int]:
    best_k, best_labels, best_score = None, None, -1
    if X.shape[0] < k_min:
        return np.zeros(X.shape[0], dtype=int), 1
    for k in range(k_min, min(k_max, X.shape[0]) + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(X)
        score = -1 if len(set(labels)) < 2 else silhouette_score(X, labels)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
    return best_labels, (best_k or 1)

# ---------------------------
# Ranking
# ---------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

def norm01(x):
    x = np.asarray(x, dtype=float)
    if np.allclose(x.max(), x.min()):
        return np.ones_like(x) * 0.5
    return (x - x.max().min())  # placeholder, will override below
# fix norm01 bug (typo-safe)
def norm01(x):
    x = np.asarray(x, dtype=float)
    if np.allclose(x.max(), x.min()):
        return np.ones_like(x) * 0.5
    return (x - x.min()) / (x.max() - x.min())

def rank_scores(
    sharp: np.ndarray,
    expo: np.ndarray,
    face_frac: np.ndarray,
    face_vecs: List[Optional[np.ndarray]],
    ref_face: Optional[np.ndarray],
    weights=(0.55, 0.20, 0.15, 0.10),
) -> np.ndarray:
    """
    Combine:
    - face similarity to reference (s_ref)
    - sharpness
    - exposure "goodness"
    - face size fraction
    """
    s_sharp = norm01(sharp)
    s_expo  = norm01(expo)
    s_face  = norm01(face_frac)

    if ref_face is not None:
        sims = []
        for fv in face_vecs:
            if fv is None: sims.append(0.3)
            else: sims.append((cosine(fv, ref_face)+1)/2)
        s_ref = np.array(sims, dtype=float)
    else:
        s_ref = np.ones(len(sharp)) * 0.5

    w1, w2, w3, w4 = weights
    final = w1*s_ref + w2*s_sharp + w3*s_expo + w4*s_face
    return final

# ---------------------------
# Pipeline
# ---------------------------
def main(args):
    # GPU for face engine; Torch uses CUDA if available & not forced CPU
    use_gpu_for_faces = args.gpu
    torch_device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

    work = Path(args.workdir); work.mkdir(exist_ok=True, parents=True)
    src = Path(args.input)

    # Collect images preserving order
    if src.suffix.lower() == ".zip":
        imgs = collect_images(src, work)
        if args.ref is None and args.ref_name:
            candidate = work / args.ref_name
            if candidate.exists():
                args.ref = str(candidate)
    else:
        imgs = collect_images(src, work)

    # Optional de-dup by perceptual hash (keeps first occurrence)
    if args.dedup:
        seen = set()
        filtered = []
        for p in imgs:
            try:
                h = imagehash.phash(Image.open(p).convert("RGB"), hash_size=16)
            except Exception:
                filtered.append(p); continue
            if h in seen: continue
            seen.add(h); filtered.append(p)
        imgs = filtered

    # Visual embedder
    embedder = GlobalEmbedder(device=torch_device)

    # Reference face
    ref_face = None
    if args.ref is not None and Path(args.ref).exists():
        pil_ref = Image.open(args.ref).convert("RGB")
        ref_face = face_embedding(pil_ref, use_gpu=use_gpu_for_faces)

    records = []
    print(f"Found {len(imgs)} images. Extracting features...")
    for idx, p in enumerate(tqdm(imgs)):
        pil = Image.open(p).convert("RGB")
        emb = embedder(pil)                    # 2048
        hist = color_hist_feature(pil)         # 96
        ph  = phash_feature(pil)               # 256
        q_sharp = sharpness_score(pil)
        q_expo  = exposure_score(pil)
        f_area  = face_box_area_fraction(pil, use_gpu=use_gpu_for_faces)
        f_vec   = face_embedding(pil, use_gpu=use_gpu_for_faces)
        feat = np.concatenate([emb, hist, ph]) # for clustering
        records.append({
            "index": idx, "path": p, "emb": emb, "feat": feat,
            "sharp": q_sharp, "expo": q_expo, "f_area": f_area, "f_vec": f_vec
        })

    # Clustering features
    X = np.stack([r["feat"] for r in records])
    pca = PCA(n_components=min(128, X.shape[1]))
    Xr = pca.fit_transform(X)

    labels, k = auto_kmeans(Xr, k_min=args.kmin, k_max=args.kmax, seed=args.seed)
    for r, lab in zip(records, labels):
        r["cluster"] = int(lab)

    # Rank within each cluster
    sets: Dict[int, List[dict]] = {}
    for r in records:
        sets.setdefault(r["cluster"], []).append(r)

    results = []
    print(f"\nAuto-detected {len(sets)} sets.")
    for lab, items in sorted(sets.items(), key=lambda x: x[0]):
        sharp = np.array([it["sharp"] for it in items])
        expos = np.array([it["expo"]  for it in items])
        farea = np.array([it["f_area"] for it in items])
        fvecs = [it["f_vec"] for it in items]

        scores = rank_scores(sharp, expos, farea, fvecs, ref_face,
                             weights=(args.w_face, args.w_sharp, args.w_expo, args.w_farea))
        for it, sc in zip(items, scores):
            it["score"] = float(sc)

        items_sorted = sorted(items, key=lambda d: d["score"], reverse=True)
        winners = items_sorted[:min(args.topk, len(items_sorted))]
        results.append((lab, winners))

    # Report to console
    print("\n=== TOP {} PER SET ===".format(args.topk))
    for lab, winners in results:
        print(f"\nSet {lab}:")
        for rank, w in enumerate(winners, 1):
            print(f"  #{rank}: index={w['index']}  file={w['path']}  score={w['score']:.3f}")

    # CSV output
    if args.csv:
        rows = []
        for lab, winners in results:
            for rank, w in enumerate(winners, 1):
                rows.append({
                    "set_id": lab,
                    "rank": rank,
                    "index": w["index"],
                    "file": str(w["path"]),
                    "score": round(w["score"], 6)
                })
        df = pd.DataFrame(rows)
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nSaved CSV: {Path(args.csv).resolve()}")

    # Optional export to folders
    if args.export and not args.zip_only:
        out_root = Path(args.export)
        if out_root.exists(): shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        for lab, winners in results:
            d = out_root / f"set_{lab:02d}"
            d.mkdir(parents=True, exist_ok=True)
            for rank, w in enumerate(winners, 1):
                dst = d / f"top{rank}_{w['path'].name}"
                shutil.copy2(w["path"], dst)
        print(f"Exported winners to: {out_root.resolve()}")

    # ZIP output (flat — all files in same folder inside the ZIP)
    if args.zip_path:
        zip_path = Path(args.zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for lab, winners in results:
                for rank, w in enumerate(winners, 1):
                    src = Path(w["path"])
                    # flat zip: prefix rank and set to avoid collisions; no subfolders
                    arcname = f"set{lab:02d}_top{rank}_{src.name}"
                    zf.write(src, arcname)
        print(f"Created ZIP of winners: {zip_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster images into sets and pick top-K per set (InsightFace GPU).")
    parser.add_argument("input", help="Folder of images OR a .zip containing images.")
    parser.add_argument("--ref", help="Optional reference face image (path).", default=None)
    parser.add_argument("--ref-name", help="If reference is inside the ZIP root (e.g., reference.jpg).", default=None)
    parser.add_argument("--export", help="Optional output folder to copy winners into (by set).", default=None)
    parser.add_argument("--csv", help="Optional CSV path to save results.", default=None)
    parser.add_argument("--workdir", default="./_img_work")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for PyTorch (ResNet features).")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for InsightFace (onnxruntime-gpu).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--dedup", action="store_true", help="Deduplicate near-identical images by pHash (keeps first).")
    parser.add_argument("--topk", type=int, default=3, help="How many winners per set to keep.")
    # scoring weights
    parser.add_argument("--w-face", type=float, default=0.55, help="Weight for face similarity to reference.")
    parser.add_argument("--w-sharp", type=float, default=0.20, help="Weight for sharpness.")
    parser.add_argument("--w-expo", type=float, default=0.15, help="Weight for exposure.")
    parser.add_argument("--w-farea", type=float, default=0.10, help="Weight for face area fraction.")
    # ZIP options
    parser.add_argument("--zip", dest="zip_path", help="Path to a ZIP to write all winners (flat, no subfolders).", default=None)
    parser.add_argument("--zip-only", action="store_true", help="Create only the ZIP (skip folder export).")
    args = parser.parse_args()
    main(args)
