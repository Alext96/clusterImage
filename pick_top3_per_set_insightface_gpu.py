import argparse, zipfile, os, shutil, warnings
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
import imagehash
import cv2
from tqdm import tqdm

# ML / optional quality features
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd

# InsightFace (GPU-capable with onnxruntime-gpu)
from insightface.app import FaceAnalysis

# ---------------------------
# Helpers / IO
# ---------------------------
IMG_EXT = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT

def read_images_from_zip(zip_path: Path, workdir: Path) -> List[Path]:
    with zipfile.ZipFile(zip_path, 'r') as zf:
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

def scenario_from_name(p: Path) -> Optional[str]:
    """
    Extract scenario ID as the filename prefix before the first hyphen.
    Example: '1855001-Alexa, ... .png' -> '1855001'
    """
    name = p.name
    if "-" not in name:
        return None
    prefix = name.split("-", 1)[0].strip()
    if prefix.isdigit() and len(prefix) >= 5:
        return prefix
    return None

# ---------------------------
# Optional visual feature extractor (only used if you turn on quality weights)
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

def sharpness_score(pil: Image.Image) -> float:
    arr = np.asarray(pil.convert("L"))
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())

def exposure_score(pil: Image.Image) -> float:
    arr = np.asarray(pil.convert("L")) / 255.0
    mu, sigma = arr.mean(), arr.std()
    return float(np.exp(-((mu-0.5)**2)/0.02) * np.exp(-((sigma-0.25)**2)/0.02))

def norm01(x):
    x = np.asarray(x, dtype=float)
    if np.allclose(x.max(), x.min()):
        return np.ones_like(x) * 0.5
    return (x - x.min()) / (x.max() - x.min())

# ---------------------------
# InsightFace: face embeddings (GPU/CPU)
# ---------------------------
_FACE_APP = None
def _get_face_app(use_gpu: bool) -> FaceAnalysis:
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

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ---------------------------
# Ranking
# ---------------------------
def rank_scores(
    face_vecs: List[Optional[np.ndarray]],
    ref_face: Optional[np.ndarray],
    sharp: Optional[np.ndarray] = None,
    expo: Optional[np.ndarray] = None,
    face_frac: Optional[np.ndarray] = None,
    w_face: float = 1.0,
    w_sharp: float = 0.0,
    w_expo: float = 0.0,
    w_farea: float = 0.0,
) -> np.ndarray:
    n = len(face_vecs)
    # Face similarity (0..1). If no ref, 0.5. If no face, small penalty.
    if ref_face is not None:
        s_ref = np.array([(cosine(v, ref_face)+1)/2 if v is not None else 0.3 for v in face_vecs], dtype=float)
    else:
        s_ref = np.ones(n, dtype=float) * 0.5

    def safe_norm(x):
        if x is None: return np.zeros(n, dtype=float)
        return norm01(x)

    s_sharp = safe_norm(sharp)
    s_expo  = safe_norm(expo)
    s_facef = safe_norm(face_frac)

    score = w_face*s_ref + w_sharp*s_sharp + w_expo*s_expo + w_farea*s_facef
    return score

# ---------------------------
# Pipeline
# ---------------------------
def main(args):
    use_gpu_for_faces = args.gpu
    torch_device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

    work = Path(args.workdir); work.mkdir(exist_ok=True, parents=True)
    src = Path(args.input)

    # Collect images preserving order
    if src.suffix.lower() == ".zip":
        imgs = collect_images(src, work)
        # If reference is inside zip root:
        if args.ref is None and args.ref_name:
            candidate = work / args.ref_name
            if candidate.exists():
                args.ref = str(candidate)
    else:
        imgs = collect_images(src, work)

    # Optional de-dup by perceptual hash (keeps first occurrence)
    if args.dedup:
        seen = set(); filtered = []
        for p in imgs:
            try:
                h = imagehash.phash(Image.open(p).convert("RGB"), hash_size=16)
            except Exception:
                filtered.append(p); continue
            if h in seen: continue
            seen.add(h); filtered.append(p)
        imgs = filtered

    # Optional (only if you turn on quality weights)
    embedder = GlobalEmbedder(device=torch_device)

    # Reference face
    ref_face = None
    if args.ref is not None and Path(args.ref).exists():
        pil_ref = Image.open(args.ref).convert("RGB")
        ref_face = face_embedding(pil_ref, use_gpu=use_gpu_for_faces)
        if ref_face is None:
            print(f"[WARN] Reference face not detected in: {args.ref}")
    else:
        print(f"[WARN] Reference not found or invalid: {args.ref}")

    # Build per-image measurements
    records = []
    print(f"Found {len(imgs)} images. Extracting face features...")
    for idx, p in enumerate(tqdm(imgs)):
        pil = Image.open(p).convert("RGB")
        f_vec = face_embedding(pil, use_gpu=use_gpu_for_faces)
        q_sharp = sharpness_score(pil) if (args.w_sharp > 0) else 0.0
        q_expo  = exposure_score(pil) if (args.w_expo  > 0) else 0.0
        f_area  = face_box_area_fraction(pil, use_gpu=use_gpu_for_faces) if (args.w_farea > 0) else 0.0
        rec = {
            "index": idx,
            "path": p,
            "scenario": scenario_from_name(p) or "misc",
            "f_vec": f_vec,
            "sharp": q_sharp,
            "expo": q_expo,
            "f_area": f_area,
        }
        records.append(rec)

    # Group by scenario prefix
    sets: Dict[str, List[dict]] = {}
    for r in records:
        sets.setdefault(r["scenario"], []).append(r)

    # Rank within each scenario
    results = []
    print(f"\nDetected {len(sets)} scenarios (by filename prefix).")
    for scen_key, items in sorted(sets.items(), key=lambda x: x[0]):
        sharp = np.array([it["sharp"] for it in items], dtype=float) if args.w_sharp > 0 else None
        expos = np.array([it["expo"]  for it in items], dtype=float) if args.w_expo  > 0 else None
        farea = np.array([it["f_area"] for it in items], dtype=float) if args.w_farea > 0 else None
        fvecs = [it["f_vec"] for it in items]

        scores = rank_scores(
            face_vecs=fvecs, ref_face=ref_face,
            sharp=sharp, expo=expos, face_frac=farea,
            w_face=args.w_face, w_sharp=args.w_sharp, w_expo=args.w_expo, w_farea=args.w_farea
        )
        for it, sc in zip(items, scores):
            it["score"] = float(sc)

        # Threshold filter
        items_keep = [it for it in items if it["score"] >= args.min_score]
        if not items_keep:
            print(f"Scenario {scen_key}: no images >= {args.min_score:.3f}, skipping.")
            winners = []
        else:
            items_sorted = sorted(items_keep, key=lambda d: d["score"], reverse=True)
            winners = items_sorted[:min(args.topk, len(items_sorted))]

        results.append((scen_key, winners))

    # Report to console
    print("\n=== TOP {} PER SCENARIO (min-score: {:.3f}) ===".format(args.topk, args.min_score))
    for scen_key, winners in results:
        if not winners:
            print(f"\nScenario {scen_key}: (no winners)")
            continue
        print(f"\nScenario {scen_key}:")
        for rank, w in enumerate(winners, 1):
            print(f"  #{rank}: index={w['index']}  file={w['path']}  score={w['score']:.3f}")

    # CSV output
    if args.csv:
        rows = []
        for scen_key, winners in results:
            for rank, w in enumerate(winners, 1):
                rows.append({
                    "scenario": scen_key,
                    "rank": rank,
                    "index": w["index"],
                    "file": str(w["path"]),
                    "score": round(w["score"], 6),
                })
        df = pd.DataFrame(rows)
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nSaved CSV: {Path(args.csv).resolve()}")

    # ZIP output (flat — all winners together, no subfolders)
    if args.zip_path:
        zip_path = Path(args.zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for scen_key, winners in results:
                for rank, w in enumerate(winners, 1):
                    src = Path(w["path"])
                    arcname = f"{scen_key}_top{rank}_{src.name}"
                    zf.write(src, arcname)
        print(f"Created ZIP of winners: {zip_path.resolve()}")

    # Optional export to a flat folder (no subfolders)
    if args.export:
        out_root = Path(args.export)
        if out_root.exists():
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        for scen_key, winners in results:
            for rank, w in enumerate(winners, 1):
                src = Path(w["path"])
                dst = out_root / f"{scen_key}_top{rank}_{src.name}"
                shutil.copy2(src, dst)
        print(f"Exported winners to: {out_root.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick top-K per scenario (filename prefix) using InsightFace GPU.")
    parser.add_argument("input", help="Folder of images OR a .zip containing images.")
    parser.add_argument("--ref", help="Reference face image path (outside the zip).", default=None)
    parser.add_argument("--ref-name", help="If the reference is inside the ZIP root (e.g., reference.png).", default=None)
    parser.add_argument("--workdir", default="./_img_work")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for PyTorch (only used if quality terms enabled).")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for InsightFace (onnxruntime-gpu).")

    parser.add_argument("--dedup", action="store_true", help="Deduplicate near-identical images by pHash (keeps first).")
    parser.add_argument("--topk", type=int, default=3, help="How many winners per scenario to keep.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop any image below this score (0..1).")

    # scoring weights (set face-only for pure likeness)
    parser.add_argument("--w-face", type=float, default=1.0, help="Weight for face similarity to reference.")
    parser.add_argument("--w-sharp", type=float, default=0.0, help="Weight for sharpness (0 to ignore).")
    parser.add_argument("--w-expo", type=float, default=0.0, help="Weight for exposure (0 to ignore).")
    parser.add_argument("--w-farea", type=float, default=0.0, help="Weight for face area fraction (0 to ignore).")

    # outputs
    parser.add_argument("--csv", help="Optional CSV path to save results.", default=None)
    parser.add_argument("--zip", dest="zip_path", help="Path to a ZIP to write all winners (flat, no subfolders).", default=None)
    parser.add_argument("--export", help="Optional output folder to copy winners into (flat, no subfolders).", default=None)
    args = parser.parse_args()
    main(args)
