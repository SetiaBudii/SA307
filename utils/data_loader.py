import os

def load_dataset(image_dir, mask_dir):
    """
    Membuat list of dicts dari pasangan path gambar dan mask.
    """
    dataset = []
    for image_name in sorted(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)
        dataset.append({
            "image": image_path,
            "annotation": mask_path
        })
    return dataset
