# source: https://github.com/zenseact/EdgeAnnotationZChallenge/blob/35afb0dcffd6b7ca3982a9a3ffbe50e9c92875f0/eval/convert_annotations_to_kitti.py#L25 
def zod_occlusion_to_kitti(zod_occlusion):
    mapping = {
        'None': 0,
        'Light': 1,
        'Medium': 1,
        'Heavy': 2,
        'VeryHeavy': 2,
        'Undefined': 2  # "If undefined we assume the worst"
    }

    return mapping.get(zod_occlusion, 3)  # 3 = unknown

def get_zod_obj_level(obj):
    height = float(obj.box2d.ymax) - float(obj.box2d.ymin) + 1 # x_min, y_min, x_max, y_max

    if height >= 40 and obj.truncation <= 0.15 and obj.occlusion <= 0:
        return 0 # Easy
    elif height >= 25 and obj.truncation <= 0.3 and obj.occlusion <= 1:
        return 1  # Moderate
    elif height >= 25 and obj.truncation <= 0.5 and obj.occlusion <= 2:
        return 2  # Hard
    else:
        return -1 # Unknown
