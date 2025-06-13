import os
import cv2
import matplotlib.pyplot as plt

image_dir = "data/homeobjects-3K/train/images"
label_dir = "data/homeobjects-3K/train/labels"

class_names = [
    "bed", "sofa", "chair", "table", "lamp", "tv", "laptop",
    "wardrobe", "window", "door", "potted plant", "photo frame"
]

#Load and display images
num_images_to_show = 3
images = sorted(os.listdir(image_dir))[:num_images_to_show]

for img_name in images:
    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    #Draw bounding boxes
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                class_id, x_center, y_center, box_w, box_h = map(float, line.strip().split())
                class_id = int(class_id)

                #YOLO to coordinate boxes
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)

                label = class_names[class_id]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    #Display
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(img_name)
    plt.axis('off')
    plt.show()
