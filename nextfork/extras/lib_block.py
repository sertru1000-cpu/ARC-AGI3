        # ---- NEXTFORK board helpers (02.10): the functions the model rewrites in most games ----
        def crop(r0, c0, r1, c1, frame=None):
            # ascii of rows r0..r1, cols c0..c1 (inclusive) with row/col headers
            f = frame if frame is not None else runtime_globals["current_frame"]
            rows = f.ascii.split("\n")
            r0, r1 = max(0, r0), min(len(rows) - 1, r1)
            out = ["     " + "".join(str((c // 10) % 10) for c in range(c0, c1 + 1)),
                   "     " + "".join(str(c % 10) for c in range(c0, c1 + 1))]
            for r in range(r0, r1 + 1):
                out.append("%3d  %s" % (r, rows[r][c0:c1 + 1]))
            text = "\n".join(out)
            print(text)

        def objects(color=None, min_pixels=1, frame=None):
            # compact object list: id color pixels (r0,c0)-(r1,c1)
            f = frame if frame is not None else runtime_globals["current_frame"]
            out = []
            for node in f.segmentation.get("nodes", []):
                if color is not None and node.get("color") != color:
                    continue
                if node.get("pixels", 0) < min_pixels:
                    continue
                rows = [p[0] for p in node.get("boundary", [])] or [0]
                cols = [p[1] for p in node.get("boundary", [])] or [0]
                out.append((node["id"], node["color"], node.get("pixels", 0), (min(rows), min(cols)), (max(rows), max(cols))))
            return out

        def bbox(color, frame=None):
            # bounding boxes (r0, c0, r1, c1) of every object of that color
            return [(a[0], a[1], b[0], b[1]) for _, _, _, a, b in objects(color, frame=frame)]

        def brief(frame=None):
            # one line: object count per color, largest object per color
            f = frame if frame is not None else runtime_globals["current_frame"]
            per = {}
            for node in f.segmentation.get("nodes", []):
                c = node.get("color")
                n, big = per.get(c, (0, 0))
                per[c] = (n + 1, max(big, node.get("pixels", 0)))
            line = " ".join("%s:%dx(max %d)" % (c, n, big) for c, (n, big) in sorted(per.items()))
            print(line)

        runtime_globals["crop"] = crop
        runtime_globals["objects"] = objects
        runtime_globals["bbox"] = bbox
        runtime_globals["brief"] = brief
        # ---- end NEXTFORK board helpers ----
