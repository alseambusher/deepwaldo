from Mask_RCNN.mrcnn import utils
import os
import json
import skimage.draw
import numpy as np

# Modified version of coco data loader from Mask-RCNN

class Data(utils.Dataset):
    def load(self, dataset_dir, subset):
        self.add_class("waldo", 1, "waldo")
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations = list(json.load(open(os.path.join(dataset_dir, "via_region_data.json"))).values())
        # skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "waldo",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "waldo":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "waldo":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)