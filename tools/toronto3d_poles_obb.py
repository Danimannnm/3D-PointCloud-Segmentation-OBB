import os
import json
import argparse
import numpy as np
from os.path import join, exists, dirname, abspath
import sys
import csv
from collections import Counter

# Ensure repository root is on sys.path for 'utils' relative imports
_SCRIPT_DIR = dirname(abspath(__file__))
_REPO_ROOT = dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import open3d as o3d
    HAS_O3D = True
    try:
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
        HAS_O3D_GUI = True
    except Exception:
        HAS_O3D_GUI = False
except Exception:
    HAS_O3D = False
    HAS_O3D_GUI = False


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
        write_ply(path, [verts], ['x','y','z'])
        return

    verts = np.vstack(verts_all).astype(np.float32)
    faces = np.vstack(faces_all).astype(np.int32)
    write_ply(path, [verts], ['x','y','z'], triangular_faces=faces)


def launch_interactive_viewer(args, xyz, pole_pts, labels, kept_points, kept_labels, obbs, min_points_keep):
    if not HAS_O3D_GUI:
        print('Open3D GUI modules not available. Install a full Open3D build (pip install open3d).')
        return

    print('Launching Open3D GUI (SceneWidget FLY mode)...')
    app = gui.Application.instance
    app.initialize()
    window = app.create_window('Toronto3D Poles Viewer', 1600, 900)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene_widget)

    # Background color (default black)
    if args.bg_color is not None and len(args.bg_color) == 3:
        rgb = [float(max(0.0, min(1.0, c))) for c in args.bg_color]
    else:
        rgb = [0.0, 0.0, 0.0]
    bg_rgba = tuple(rgb + [1.0])
    scene_widget.scene.set_background(bg_rgba)

    # Print quick instruction for users; FLY controls remain default.
    print('Interactive viewer controls: FLY navigation (WASD + mouse). Press R to level camera to world-up.')
    def make_point_material():
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = float(args.point_size)
        return mat

    def make_line_material():
        mat = rendering.MaterialRecord()
        mat.shader = 'unlitLine'
        mat.line_width = 1.5
        return mat

    def make_mesh_material(color=(0.1, 0.6, 1.0)):
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultLit'
        mat.base_color = (*color, 1.0)
        return mat

    bbox = None

    def accumulate_bbox(geometry):
        nonlocal bbox
        b = geometry.get_axis_aligned_bounding_box()
        if bbox is None:
            bbox = b
        else:
            min_bound = np.minimum(bbox.get_min_bound(), b.get_min_bound())
            max_bound = np.maximum(bbox.get_max_bound(), b.get_max_bound())
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # Base geometry
    if args.show_base != 'none':
        if args.show_base == 'all':
            base_xyz = xyz
        elif args.show_base == 'poles':
            base_xyz = kept_points
        elif args.show_base == 'subsample':
            tmp = o3d.geometry.PointCloud()
            tmp.points = o3d.utility.Vector3dVector(xyz)
            tmp = tmp.voxel_down_sample(voxel_size=args.voxel)
            base_xyz = np.asarray(tmp.points)
        else:
            base_xyz = xyz

        if base_xyz.shape[0] > args.max_view_points:
            sel = np.random.default_rng(0).choice(base_xyz.shape[0], size=args.max_view_points, replace=False)
            base_xyz = base_xyz[sel]

        if base_xyz.shape[0] > 0:
            base_pc = o3d.geometry.PointCloud()
            base_pc.points = o3d.utility.Vector3dVector(base_xyz)
            base_pc.colors = o3d.utility.Vector3dVector(np.full((base_xyz.shape[0], 3), 0.7, dtype=np.float64))
            scene_widget.scene.add_geometry('base_cloud', base_pc, make_point_material())
            accumulate_bbox(base_pc)

    # Pole clusters point cloud
    if kept_points.shape[0] > 0:
        cluster_pc = o3d.geometry.PointCloud()
        cluster_pc.points = o3d.utility.Vector3dVector(kept_points)
        max_label = kept_labels.max() if kept_labels.size > 0 else -1
        rng = np.random.default_rng(42)
        lut = rng.uniform(0.0, 1.0, size=(max_label + 1, 3)) if max_label >= 0 else np.zeros((1, 3))
        cols = np.zeros((kept_points.shape[0], 3), dtype=np.float32)
        for cid in np.unique(kept_labels):
            cols[kept_labels == cid] = lut[int(cid)]
        cluster_pc.colors = o3d.utility.Vector3dVector(cols)
        scene_widget.scene.add_geometry('pole_clusters', cluster_pc, make_point_material())
        accumulate_bbox(cluster_pc)

    # OBB lines
    if not args.no_obb:
        edges = np.array([
            [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
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
            scene_widget.scene.add_geometry(f"obb_{int(obb['id'])}", ls, make_line_material())
            accumulate_bbox(ls)

    # Cylinders (optional)
    if args.cylinder_mesh:
        for cid in np.unique(labels):
            if cid < 0:
                continue
            cluster_mask = labels == cid
            pts = pole_pts[cluster_mask]
            if pts.shape[0] < min_points_keep:
                continue
            zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
            h = zmax - zmin
            if h <= 0:
                continue
            center = np.array([pts[:, 0].mean(), pts[:, 1].mean(), 0.5 * (zmin + zmax)])
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=args.cyl_radius, height=h, resolution=args.cyl_segments, split=1)
            mesh.translate(center - np.array([0, 0, h / 2]))
            mesh.paint_uniform_color([0.1, 0.6, 1.0])
            scene_widget.scene.add_geometry(f'cyl_{int(cid)}', mesh, make_mesh_material())
            accumulate_bbox(mesh)

    if bbox is not None:
        scene_widget.setup_camera(60.0, bbox, bbox.get_center())
        default_center = bbox.get_center()
        default_diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    else:
        default_center = np.zeros(3, dtype=np.float32)
        default_diag = 10.0

    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def realign_camera_to_world():
        camera = scene_widget.scene.camera
        center = camera.get_center() if hasattr(camera, 'get_center') else default_center
        eye = camera.get_eye() if hasattr(camera, 'get_eye') else (default_center + np.array([0.0, -default_diag, default_diag]))
        forward = center - eye
        dist = np.linalg.norm(forward)
        if dist < 1e-6:
            dist = max(default_diag, 1.0)
            forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            forward = forward / dist
        right = np.cross(forward, up_world)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            right_norm = 1.0
        right = right / right_norm
        forward = np.cross(up_world, right)
        forward = forward / np.linalg.norm(forward)
        if bbox is not None:
            center = bbox.get_center()
            dist = max(dist, default_diag * 0.75)
        new_eye = center - forward * dist
        if hasattr(camera, 'look_at'):
            camera.look_at(center, new_eye, up_world)
        else:
            scene_widget.setup_camera(60.0, bbox if bbox is not None else o3d.geometry.AxisAlignedBoundingBox(center - dist, center + dist), center)

    try:
        key_realign = gui.KeyName.R
    except AttributeError:
        key_realign = ord('R')

    try:
        EventResult = gui.EventCallbackResult
        event_handled = lambda: EventResult.HANDLED
        event_ignored = lambda: EventResult.IGNORED
    except AttributeError:
        EventResult = None
        event_handled = lambda: True
        event_ignored = lambda: False

    def on_key(event):
        try:
            event_type = event.type
            event_key = event.key
        except AttributeError:
            return event_ignored()
        down_type = getattr(gui.KeyEvent.Type, 'DOWN', None)
        if down_type is not None and event_type == down_type and event_key == key_realign:
            realign_camera_to_world()
            return event_handled()
        return event_ignored()

    window.set_on_key(on_key)

    def on_layout(layout_context):
        scene_widget.frame = window.content_rect

    window.set_on_layout(on_layout)

    scene_widget.set_view_controls(gui.SceneWidget.Controls.FLY)

    app.run()


def main():
    parser = argparse.ArgumentParser(description='Toronto3D Poles OBB extraction')
    parser.add_argument('--test_log', type=str, required=True, help='Path to test/<Log_...> directory')
    parser.add_argument('--probs_name', type=str, default='L002.ply', help='Base name of probs ply (default L002.ply)')
    parser.add_argument('--probs_glob', type=str, default=None, help='Optional glob relative to test_log to locate probs (overrides probs_name)')
    parser.add_argument('--pole_label', type=str, default='Pole', help='Class name for poles (prob column)')
    parser.add_argument('--pole_thresh', type=float, default=0.5, help='Probability threshold for pole mask')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN eps (meters)')
    parser.add_argument('--min_samples', type=int, default=15, help='DBSCAN min_samples')
    parser.add_argument('--min_points', type=int, default=30, help='Discard clusters smaller than this many points when fitting OBBs')
    parser.add_argument('--min_points_keep', type=int, default=None, help='Discard clusters smaller than this many points entirely (defaults to min_points)')
    parser.add_argument('--min_height', type=float, default=0.0, help='Discard clusters shorter than this z-height (meters)')
    parser.add_argument('--max_height', type=float, default=None, help='Discard clusters taller than this z-height (meters)')
    parser.add_argument('--min_mean_prob', type=float, default=0.0, help='Discard clusters whose mean pole probability is below this threshold')
    parser.add_argument('--show', action='store_true', help='Show Open3D visualization with OBB overlay')
    parser.add_argument('--show_base', type=str, default='poles', choices=['all', 'poles', 'none', 'subsample'],
                        help="What to display as base: full scene ('all'), only pole points ('poles'), disable base ('none'), or voxel subsample ('subsample')")
    parser.add_argument('--voxel', type=float, default=0.10, help='Voxel size (m) for base subsampling when show_base=subsample')
    parser.add_argument('--max_view_points', type=int, default=1000000, help='Cap on number of points sent to Open3D per cloud')
    parser.add_argument('--no_obb', action='store_true', help='Skip OBB computation and export (just highlight poles)')
    parser.add_argument('--cylinder_mesh', action='store_true', help='Approximate each pole cluster with a vertical cylinder mesh in visualization')
    parser.add_argument('--cyl_radius', type=float, default=0.15, help='Cylinder radius approximation (meters)')
    parser.add_argument('--cyl_segments', type=int, default=16, help='Number of radial segments for cylinder mesh')
    parser.add_argument('--print_stats', action='store_true', help='Print per-cluster statistics (height, points, OBB extents)')
    parser.add_argument('--export_csv', action='store_true', help='Export per-cluster statistics as CSV alongside JSON')
    parser.add_argument('--interactive', action='store_true', help='Launch Open3D SceneWidget viewer (FLY mode)')
    parser.add_argument('--point_size', type=float, default=2.0, help='Point size for interactive viewer points')
    parser.add_argument('--bg_color', type=float, nargs=3, default=None, help='Background RGB (0-1 each) for interactive viewer (default black)')
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

    base_name = os.path.splitext(args.probs_name)[0]

    print(f'Loading probabilities: {probs_path}')
    xyz, probs, prob_keys = load_probs_ply(probs_path)

    if args.pole_label not in prob_keys:
        raise ValueError(f'Pole label "{args.pole_label}" not found in prob columns: {prob_keys}')
    pole_col = prob_keys.index(args.pole_label)
    pole_prob = probs[:, pole_col]
    mask = pole_prob >= args.pole_thresh
    pole_pts = xyz[mask]
    pole_probs_masked = pole_prob[mask]
    print(f'Pole mask points: {pole_pts.shape[0]} / {xyz.shape[0]} (threshold={args.pole_thresh})')

    if pole_pts.shape[0] == 0:
        print('No pole points above threshold. Exiting.')
        return

    print(f'Clustering with DBSCAN: eps={args.eps}, min_samples={args.min_samples}')
    labels = dbscan_cluster(pole_pts, eps=args.eps, min_samples=args.min_samples)
    labels_orig = labels.copy()
    valid_orig = labels_orig >= 0
    if valid_orig.sum() == 0:
        print('No clusters found. Exiting.')
        return

    clusters_ply = join(args.test_log, f'poles_clusters_{base_name}.ply')

    cluster_stats = []
    obbs = []
    obb_kept = 0
    labels_filtered = labels.copy()
    unique_clusters = np.unique(labels_orig[valid_orig])
    total_clusters = len(unique_clusters)
    min_points_keep = args.min_points_keep if args.min_points_keep is not None else args.min_points
    filtered_records = []

    for cid in unique_clusters:
        mask_cluster = (labels_orig == cid)
        pts = pole_pts[mask_cluster]
        if pts.shape[0] == 0:
            continue
        zmin = float(pts[:, 2].min())
        zmax = float(pts[:, 2].max())
        height_z = float(zmax - zmin)
        centroid = pts.mean(axis=0).astype(np.float32)
        cluster_probs = pole_probs_masked[mask_cluster]
        mean_prob = float(cluster_probs.mean()) if cluster_probs.size > 0 else 0.0

        stat = {
            'id': int(cid),
            'num_points': int(pts.shape[0]),
            'height_z': height_z,
            'z_min': zmin,
            'z_max': zmax,
            'centroid': centroid.tolist(),
            'mean_prob': mean_prob,
            'has_obb': False
        }

        keep = True
        reasons = []
        if stat['num_points'] < min_points_keep:
            keep = False
            reasons.append('min_points_keep')
        if height_z < args.min_height:
            keep = False
            reasons.append('min_height')
        if args.max_height is not None and height_z > args.max_height:
            keep = False
            reasons.append('max_height')
        if mean_prob < args.min_mean_prob:
            keep = False
            reasons.append('min_mean_prob')

        if not keep:
            labels_filtered[mask_cluster] = -1
            filtered_records.append({'id': int(cid), 'reasons': reasons, 'num_points': stat['num_points'], 'height_z': height_z, 'mean_prob': mean_prob})
            continue

        if not args.no_obb and pts.shape[0] >= args.min_points:
            c, ext, R = pca_obb(pts)
            obbs.append({
                'id': int(cid),
                'center': c.tolist(),
                'extents': ext.tolist(),
                'R': R.tolist(),
                'num_points': int(pts.shape[0]),
                'height_z': height_z,
                'centroid': centroid.tolist(),
                'z_min': zmin,
                'z_max': zmax
            })
            stat.update({
                'has_obb': True,
                'obb_center': c.tolist(),
                'obb_extents': ext.tolist(),
                'obb_R': R.tolist(),
                'obb_volume': float(ext[0] * ext[1] * ext[2])
            })
            obb_kept += 1
        else:
            stat['too_small_for_obb'] = pts.shape[0] < args.min_points

        cluster_stats.append(stat)

    labels = labels_filtered
    valid = labels >= 0

    kept_clusters = len(cluster_stats)
    print(f'Clusters kept after filtering: {kept_clusters} / {total_clusters} (min_height={args.min_height}, min_mean_prob={args.min_mean_prob}, min_points_keep={min_points_keep})')
    if args.max_height is not None:
        print(f'Max height filter applied: {args.max_height} m')
    if filtered_records:
        reason_counts = Counter(reason for rec in filtered_records for reason in rec['reasons'])
        reason_str = ', '.join(f"{key}:{value}" for key, value in sorted(reason_counts.items()))
        print(f'Clusters filtered out: {len(filtered_records)} ({reason_str})')

    try:
        save_clusters_ply(clusters_ply, pole_pts, np.where(valid, labels, -1))
        print(f'Cluster preview saved: {clusters_ply}')
    except Exception as e:
        print(f'Could not save cluster PLY: {e}')

    kept_mask = valid
    kept_points = pole_pts[kept_mask]
    kept_labels = labels[kept_mask]

    if not args.no_obb:
        print(f'OBBs computed: {obb_kept} clusters kept (min_points={args.min_points})')
    else:
        print('Skipping OBB computation (--no_obb).')

    dims_json = join(args.test_log, f'poles_dims_{base_name}.json')
    with open(dims_json, 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    print(f'Cluster dimensions JSON saved: {dims_json}')

    if args.export_csv:
        csv_path = join(args.test_log, f'poles_dims_{base_name}.csv')
        fieldnames = [
            'id', 'num_points', 'height_z', 'z_min', 'z_max',
            'centroid_x', 'centroid_y', 'centroid_z', 'mean_prob',
            'has_obb', 'obb_extent_x', 'obb_extent_y', 'obb_extent_z', 'obb_volume'
        ]
        with open(csv_path, 'w', newline='') as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for stat in cluster_stats:
                row = {
                    'id': stat['id'],
                    'num_points': stat['num_points'],
                    'height_z': stat['height_z'],
                    'z_min': stat['z_min'],
                    'z_max': stat['z_max'],
                    'centroid_x': stat['centroid'][0],
                    'centroid_y': stat['centroid'][1],
                    'centroid_z': stat['centroid'][2],
                    'mean_prob': stat['mean_prob'],
                    'has_obb': stat['has_obb'],
                    'obb_extent_x': '',
                    'obb_extent_y': '',
                    'obb_extent_z': '',
                    'obb_volume': ''
                }
                if stat.get('has_obb'):
                    row.update({
                        'obb_extent_x': stat['obb_extents'][0],
                        'obb_extent_y': stat['obb_extents'][1],
                        'obb_extent_z': stat['obb_extents'][2],
                        'obb_volume': stat.get('obb_volume', '')
                    })
                writer.writerow(row)
        print(f'Cluster dimensions CSV saved: {csv_path}')

    if args.print_stats:
        print('\nCluster statistics:')
        header = f"{'ID':>4} {'Points':>8} {'Height(m)':>10} {'MeanProb':>9} {'OBB?':>5}"
        print(header)
        print('-' * len(header))
        for stat in cluster_stats:
            print(f"{stat['id']:>4} {stat['num_points']:>8} {stat['height_z']:>10.3f} {stat['mean_prob']:>9.3f} {('Y' if stat['has_obb'] else 'N'):>5}")

    if not args.no_obb and obbs:
        # Save OBB JSON
        obb_json = join(args.test_log, f'poles_obb_{base_name}.json')
        save_obb_json(obb_json, obbs)
        print(f'OBB JSON saved: {obb_json}')

        # Save OBB mesh
        obb_mesh = join(args.test_log, f'poles_obb_{base_name}.ply')
        try:
            save_obb_mesh_ply(obb_mesh, obbs)
            print(f'OBB mesh saved: {obb_mesh}')
        except Exception as e:
            print(f'Could not save OBB mesh PLY: {e}')

    if args.interactive:
        launch_interactive_viewer(args, xyz, pole_pts, labels, kept_points, kept_labels, obbs, min_points_keep)
        return

    # Optional visualization
    if args.show:
        if not HAS_O3D:
            print('Open3D not installed; cannot show visualization. pip install open3d')
            return
        print('Launching Open3D viewer...')
        geoms = []
        kept_mask = valid
        kept_points = pole_pts[kept_mask]
        kept_labels = labels[kept_mask]
        # Base cloud options
        if args.show_base != 'none':
            if args.show_base == 'all':
                base_xyz = xyz
            elif args.show_base == 'poles':
                base_xyz = kept_points
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
        if kept_points.shape[0] > 0:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(kept_points)
            max_label = kept_labels.max() if kept_labels.size > 0 else -1
            rng = np.random.default_rng(42)
            lut = rng.uniform(0.0, 1.0, size=(max_label + 1, 3)) if max_label >= 0 else np.zeros((1, 3))
            cols = np.zeros((kept_points.shape[0], 3), dtype=np.float32)
            for cid in np.unique(kept_labels):
                cols[kept_labels == cid] = lut[int(cid)]
            pc.colors = o3d.utility.Vector3dVector(cols)
            geoms.append(pc)

        if not args.no_obb:
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

        if args.cylinder_mesh:
            # Approximate each cluster as vertical cylinder based on PCA vertical extent
            for cid in np.unique(labels[valid]):
                pts = pole_pts[labels == cid]
                if pts.shape[0] < min_points_keep:
                    continue
                # Height from z-range
                zmin, zmax = pts[:,2].min(), pts[:,2].max()
                h = zmax - zmin
                if h <= 0:
                    continue
                center = np.array([pts[:,0].mean(), pts[:,1].mean(), 0.5*(zmin+zmax)])
                mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=args.cyl_radius, height=h, resolution=args.cyl_segments, split=1)
                mesh.translate(center - np.array([0,0,h/2]))
                mesh.paint_uniform_color([0.1,0.6,1.0])
                geoms.append(mesh)
        o3d.visualization.draw(geoms)


if __name__ == '__main__':
    main()
