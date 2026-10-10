import cv2
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")

image_path = "/belle_bot/vision/pose_estimation/bus.jpg"
im = cv2.imread(image_path)[:, :, ::-1].copy()
print(im.dtype)

results = model(image_path)


if __name__ == "__main__":
    # Access the results
    for result in results:
        xy = result.keypoints.xy  # x and y coordinates
        xyn = result.keypoints.xyn  # normalized
        kpts = result.keypoints.data  # x, y, visibility (if available)

        print(dir(result.keypoints))
        print(dir(result))

        result = result.keypoints.data.numpy()
        person_count = result.shape[0]

        for person in range(person_count):
            m = result[person]
            for kp in list(m):
                im = cv2.circle(
                    im,
                    (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

        print("---")
        print(xy.shape)
        print(xyn.data)
        print(xyn.shape)
        print(kpts.shape)

    cv2.imshow("", im[:, :, ::-1])
    cv2.waitKey(0)


