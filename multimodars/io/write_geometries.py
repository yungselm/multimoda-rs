import math

def centerline_to_obj(cl, filename: str) -> None:
    """
    Write out a centerline as an OBJ with:
      - vertex positions (v x y z)
      - vertex normals (vn nx ny nz), if normals are set
      - a single poly-line (l 1 2 3 ... N)

    Args:
        cl:        A PyCenterline instance
        filename:  Path to write (e.g. "my_centerline.obj")
    """
    from ..multimodars import PyCenterline

    if not isinstance(cl, PyCenterline):
        raise TypeError("Expected PyCenterline instance")
    
    with open(filename, "w") as f:
        # write only _finite_ vertices
        good_pts = []
        for i, pt in enumerate(cl.points):
            x,y,z = pt.contour_point.x, pt.contour_point.y, pt.contour_point.z
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                # skip any malformed point
                continue
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            good_pts.append(pt)

        # now normals (if you still want them)
        has_normals = any(
            math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz)
            for pt in good_pts
            for nx,ny,nz in [pt.normal]
        )
        if has_normals:
            for pt in good_pts:
                nx,ny,nz = pt.normal
                if math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz):
                    f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
                else:
                    # you can either skip or write a default (0,0,0)
                    f.write("vn 0.000000 0.000000 0.000000\n")

        # stitch the filtered points
        idxs = " ".join(str(i+1) for i in range(len(good_pts)))
        f.write(f"l {idxs}\n")
    print(f"Wrote {len(good_pts)} valid points to {filename!r}")
