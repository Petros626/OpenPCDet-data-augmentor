def zod_occlusion_to_kitti(zod_occlusion): 
    mapping = {
        'None': 0,
        'Light': 1,
        'Medium': 1,
        'Heavy': 2,
        'VeryHeavy': 2
    }

    return mapping.get(zod_occlusion, 3)  # 3 = unknown

def get_zod_obj_level(obj):
    height = float(obj.box2d.ymax) - float(obj.box2d.ymin) + 1

    if height >= 40 and obj.truncation <= 0.15 and obj.occlusion <= 0:
        return 0 # Easy
    elif height >= 25 and obj.truncation <= 0.3 and obj.occlusion <= 1:
        return 1  # Moderate
    elif height >= 25 and obj.truncation <= 0.5 and obj.occlusion <= 2:
        return 2  # Hard
    else:
        return -1 # Unknown
