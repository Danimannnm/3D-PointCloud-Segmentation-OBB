import os
import json
import argparse
import numpy as np
from os.path import join, exists, dirname, abspath
import sys

# Ensure repository root is on sys.path for 'utils' relative imports
_SCRIPT_DIR = dirname(abspath(__file__))
_REPO_ROOT = dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False


def load_probs_ply(ply_path):
    # Minimal PLY reader for float probabilities; rely on utils.ply if available
    from utils.ply import read_ply
    data = read_ply(ply_path)
    xyz = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
    # Infer prob columns (all not x,y,z)
    keys = list(data.dtype.names)
    prob_keys = [k for k in keys if k not in ('x', 'y', 'z')]
    probs = np.vstack([data[k] for k in prob_keys]).T.astype(np.float32)
    return xyz, probs, prob_keys


def dbscan_cluster(points, eps=0.5, min_samples=15):
    try:
        from sklearn.cluster import DBSCAN
    except Exception as e:
        raise RuntimeError('scikit-learn is required for DBSCAN clustering. Please install scikit-learn.')
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    return labels


def pca_obb(points):
    # Compute PCA axes
    pts = points.astype(np.float64)
    c = pts.mean(axis=0)
    X = pts - c
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    # Sort by descending variance
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    # Project points to PCA frame and get extents
    proj = X @ vecs
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    extents = maxs - mins
    center_local = 0.5 * (maxs + mins)
    center_world = c + vecs @ center_local
    R = vecs  # columns are axes
    return center_world.astype(np.float32), extents.astype(np.float32), R.astype(np.float32)


def save_clusters_ply(path, points, labels):
    # Save colored clusters for quick inspection
    from utils.ply import write_ply
    max_label = labels.max()
    colors = np.zeros_like(points, dtype=np.uint8)
    rng = np.random.default_rng(42)
    lut = rng.integers(0, 255, size=(max_label + 1, 3), dtype=np.uint8) if max_label >= 0 else np.zeros((1, 3), dtype=np.uint8)
    for li in range(max_label + 1):
        colors[labels == li] = lut[li]
    write_ply(path, [points.astype(np.float32), colors], ['x', 'y', 'z', 'red', 'green', 'blue'])


def save_obb_json(path, obbs):
    with open(path, 'w') as f:
        json.dump(obbs, f, indent=2)


def save_obb_mesh_ply(path, obbs):
    from utils.ply import write_ply
    verts_all = []
    faces_all = []
    v_offset = 0
    for obb in obbs:
        c = np.array(obb['center'], dtype=np.float32)
        ext = np.array(obb['extents'], dtype=np.float32)
        R = np.array(obb['R'], dtype=np.float32)
        half = 0.5 * ext
        # 8 corners in local frame
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float32)
        corners_local = signs * half
        corners_world = (R @ corners_local.T).T + c
        verts_all.append(corners_world)
        # 12 triangles composing a box
        faces = np.array([
            [0,1,3],[0,3,2],
            [4,6,7],[4,7,5],
            [0,4,5],[0,5,1],
            [2,3,7],[2,7,6],
            [0,2,6],[0,6,4],
            [1,5,7],[1,7,3]
        ], dtype=np.int32) + v_offset
        faces_all.append(faces)
        v_offset += 8
    if len(verts_all) == 0:
        verts = np.zeros((0,3), dtype=np.float32)
        faces = np.zeros((0,3), dtype=np.int32)
    else:
        verts = np.vstack(verts_all).astype(np.float32)
        faces = np.vstack(faces_all).astype(np.int32)
    write_ply(path, [verts, faces], ['x','y','z','tri1','tri2','tri3'], as_text=False)


def main():
    parser = argparse.ArgumentParser(description='Toronto3D Poles OBB extraction')
    parser.add_argument('--test_log', type=str, required=True, help='Path to test/<Log_...> directory')
    parser.add_argument('--probs_name', type=str, default='L002.ply', help='Base name of probs ply (default L002.ply)')
    parser.add_argument('--probs_glob', type=str, default=None, help='Optional glob relative to test_log to locate probs (overrides probs_name)')
    parser.add_argument('--pole_label', type=str, default='Pole', help='Class name for poles (prob column)')
    parser.add_argument('--pole_thresh', type=float, default=0.5, help='Probability threshold for pole mask')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN eps (meters)')
    parser.add_argument('--min_samples', type=int, default=15, help='DBSCAN min_samples')
    parser.add_argument('--min_points', type=int, default=30, help='Discard clusters smaller than this many points')
    parser.add_argument('--show', action='store_true', help='Show Open3D visualization with OBB overlay')
    parser.add_argument('--show_base', type=str, default='poles', choices=['all', 'poles', 'none', 'subsample'],
                        help="What to display as base: full scene ('all'), only pole points ('poles'), disable base ('none'), or voxel subsample ('subsample')")
    parser.add_argument('--voxel', type=float, default=0.10, help='Voxel size (m) for base subsampling when show_base=subsample')
    parser.add_argument('--max_view_points', type=int, default=1000000, help='Cap on number of points sent to Open3D per cloud')
    args = parser.parse_args()

    # Resolve probability file path
    probs_path = None
    if args.probs_glob:
        import glob
        matches = glob.glob(join(args.test_log, args.probs_glob), recursive=True)
        if len(matches) == 0:
            raise FileNotFoundError(f'No files matched probs_glob pattern: {args.probs_glob}')
        if len(matches) > 1:
            # Prefer one with /probs/ in path
            probs_candidates = [m for m in matches if '\\probs\\' in m or '/probs/' in m]
            if len(probs_candidates) == 1:
                probs_path = probs_candidates[0]
            else:
                probs_path = matches[0]
        else:
            probs_path = matches[0]
    else:
        direct = join(args.test_log, 'probs', args.probs_name)
        if exists(direct):
            probs_path = direct
        else:
            # Try nested dataset structure e.g. probs/Toronto3D/train/<name>
            candidate = join(args.test_log, 'probs', 'Toronto3D', 'train', args.probs_name)
            if exists(candidate):
                probs_path = candidate
            else:
                # Fallback: recursive search for filename
                import glob
                matches = glob.glob(join(args.test_log, '**', args.probs_name), recursive=True)
                probs_path = None
                for m in matches:
                    if ('/probs/' in m or '\\probs\\' in m) and m.endswith(args.probs_name):
                        probs_path = m
                        break
                if probs_path is None:
                    raise FileNotFoundError(f'Probs PLY not found. Tried: {direct}, {candidate}, recursive under {args.test_log}')

    print(f'Loading probabilities: {probs_path}')
    xyz, probs, prob_keys = load_probs_ply(probs_path)

    if args.pole_label not in prob_keys:
        raise ValueError(f'Pole label "{args.pole_label}" not found in prob columns: {prob_keys}')
    pole_col = prob_keys.index(args.pole_label)
    pole_prob = probs[:, pole_col]
    mask = pole_prob >= args.pole_thresh
    pole_pts = xyz[mask]
    print(f'Pole mask points: {pole_pts.shape[0]} / {xyz.shape[0]} (threshold={args.pole_thresh})')

    if pole_pts.shape[0] == 0:
        print('No pole points above threshold. Exiting.')
        return

    print(f'Clustering with DBSCAN: eps={args.eps}, min_samples={args.min_samples}')
    labels = dbscan_cluster(pole_pts, eps=args.eps, min_samples=args.min_samples)
    valid = labels >= 0
    if valid.sum() == 0:
        print('No clusters found. Exiting.')
        return

    # Save clusters preview
    clusters_ply = join(args.test_log, 'poles_clusters_L002.ply')
    try:
        save_clusters_ply(clusters_ply, pole_pts, np.where(valid, labels, -1))
        print(f'Cluster preview saved: {clusters_ply}')
    except Exception as e:
        print(f'Could not save cluster PLY: {e}')

    # Compute OBBs
    obbs = []
    for cid in np.unique(labels[valid]):
        pts = pole_pts[labels == cid]
        if pts.shape[0] < args.min_points:
            continue
        c, ext, R = pca_obb(pts)
        obbs.append({
            'id': int(cid),
            'center': c.tolist(),
            'extents': ext.tolist(),
            'R': R.tolist(),
            'num_points': int(pts.shape[0])
        })
    print(f'OBBs computed: {len(obbs)} clusters kept (min_points={args.min_points})')

    # Save OBB JSON
    obb_json = join(args.test_log, 'poles_obb_L002.json')
    save_obb_json(obb_json, obbs)
    print(f'OBB JSON saved: {obb_json}')

    # Save OBB mesh
    obb_mesh = join(args.test_log, 'poles_obb_L002.ply')
    try:
        save_obb_mesh_ply(obb_mesh, obbs)
        print(f'OBB mesh saved: {obb_mesh}')
    except Exception as e:
        print(f'Could not save OBB mesh PLY: {e}')

    # Optional visualization
    if args.show:
        if not HAS_O3D:
            print('Open3D not installed; cannot show visualization. pip install open3d')
            return
        print('Launching Open3D viewer...')
        geoms = []
        # Base cloud options
        if args.show_base != 'none':
            if args.show_base == 'all':
                base_xyz = xyz
            elif args.show_base == 'poles':
                base_xyz = pole_pts
            elif args.show_base == 'subsample':
                # voxel subsample full scene
                tmp = o3d.geometry.PointCloud()
                tmp.points = o3d.utility.Vector3dVector(xyz)
                tmp = tmp.voxel_down_sample(voxel_size=args.voxel)
                base_xyz = np.asarray(tmp.points)
            else:
                base_xyz = xyz

            # Cap number of points for viewer
            if base_xyz.shape[0] > args.max_view_points:
                sel = np.random.default_rng(0).choice(base_xyz.shape[0], size=args.max_view_points, replace=False)
                base_xyz = base_xyz[sel]

            if base_xyz.shape[0] > 0:
                base = o3d.geometry.PointCloud()
                base.points = o3d.utility.Vector3dVector(base_xyz)
                base.colors = o3d.utility.Vector3dVector(np.full((base_xyz.shape[0], 3), 0.7, dtype=np.float64))
                geoms.append(base)

        # Pole clusters colored
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pole_pts)
        max_label = labels[valid].max() if valid.any() else -1
        rng = np.random.default_rng(42)
        lut = rng.uniform(0.0, 1.0, size=(max_label + 1, 3)) if max_label >= 0 else np.zeros((1, 3))
        cols = np.zeros((pole_pts.shape[0], 3), dtype=np.float32)
        for cid in np.unique(labels[valid]):
            cols[labels == cid] = lut[int(cid)]
        pc.colors = o3d.utility.Vector3dVector(cols)
        geoms.append(pc)

        # OBB line sets
        edges = np.array([
            [0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]
        ], dtype=np.int32)
        for obb in obbs:
            c = np.array(obb['center'])
            ext = np.array(obb['extents'])
            R = np.array(obb['R'])
            box = o3d.geometry.OrientedBoundingBox(center=c, R=R, extent=ext)
            pts = np.asarray(box.get_box_points())
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts),
                lines=o3d.utility.Vector2iVector(edges)
            )
            ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.1, 0.9, 0.1]]), (edges.shape[0], 1)))
            geoms.append(ls)
        o3d.visualization.draw(geoms)


if __name__ == '__main__':
    main()
