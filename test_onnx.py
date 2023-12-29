import torch
import onnxruntime
import cv2
import torchvision.transforms as torchvision_T
import numpy as np

BUFFER = 10
IMAGE_SIZE = 384

def find_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    m1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    b1 = y1 - m1 * x1

    m2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else float('inf')
    b2 = y3 - m2 * x3

    x = (b2 - b1) / (m1 - m2) if (m1 - m2) != 0 else float('inf')
    y = m1 * x + b1

    return [int(x), int(y)]

def order_points(pts):
    """Rearrange coordinates to order:
    top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype("int").tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)

mean=(0.4611, 0.4359, 0.3905)
std=(0.2193, 0.2150, 0.2109)
common_transforms = torchvision_T.Compose(
        [
            torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean, std),
        ]
    )

image_path = "/home/aittgp/vutt/workspace/Document-Scanner/Epay_AI/IMG_20231214_110625.jpg"
# image_path = "/home/aittgp/vutt/workspace/Document-Scanner/Epay_Output/IMG_20231214_110122.jpg"
image_path = "/home/aittgp/vutt/workspace/Document-Scanner/Epay_AI/IMG_20231214_105544.jpg"
# image_path = "./test3.jpg"
image = cv2.imread(image_path)
h, w = image.shape[:2]
IMAGE_SIZE = 384
# IMAGE_SIZE = image_size
half = IMAGE_SIZE // 2
print(half)
imH, imW, C = image.shape
print(image.shape)

image_model = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

cv2.imwrite("resize.jpg", image_model)
scale_x = imW / IMAGE_SIZE
scale_y = imH / IMAGE_SIZE
print(f"scale: {scale_x}, {scale_y}")

# for wrap.onnx
# image_model = common_transforms(image_model)

image_model = np.asarray(image_model.transpose((2,0,1))/255.0, dtype=np.float32)

print(image_model.shape)
print(image_model.dtype)
# ==============================================================


# sess = onnxruntime.InferenceSession('./wrap.onnx',providers=['CPUExecutionProvider'])
sess = onnxruntime.InferenceSession('./mbv3_pre_wrap_uint8.onnx',providers=['CPUExecutionProvider'])

inputs_name = [x.name for x in sess.get_inputs()]
outputs_name = [x.name for x in sess.get_outputs()]
print(inputs_name)
print(outputs_name)

inputs_shape = [x.shape for x in sess.get_inputs()]
outputs_shape = [x.shape for x in sess.get_outputs()]
print(inputs_shape)
print(outputs_shape)

out = sess.run(outputs_name, {sess.get_inputs()[0].name: np.expand_dims(image_model, axis=0)})[0]

print(out)
print(out.shape)
# print(type(out))
# print(out.dtype)
# print("===============")
# for o in out:
#     print(o)
# ==============================================================

r_H, r_W = out.shape


_out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
print(_out_extended.shape)
_out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
out = _out_extended.copy()
cv2.imwrite("out.jpg", out)
# Edge Detection.
canny = cv2.Canny(out.astype(np.uint8), 225, 255)
canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f"contours: {len(contours)}")
print(contours[0].shape, contours[1].shape)
page = sorted(contours, key=cv2.contourArea, reverse=True)[0]
print(f"page: {page} {len(page)}")
# ==========================================
epsilon = 0.02 * cv2.arcLength(page, True)
print(epsilon)
corners = cv2.approxPolyDP(page, epsilon, True)

corners = np.concatenate(corners).astype(np.float32)
# print(f"coners {corners}, {corners.shape}")
corners[:, 0] -= half
corners[:, 1] -= half
# print(f"coners {corners}, {corners.shape}")

corners[:, 0] *= scale_x
corners[:, 1] *= scale_y

print(corners)
# print(f"coners {corners}, {corners.shape}")
# check if corners are inside.
# if not find smallest enclosing box, expand_image then extract document
# else extract document

if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):

    left_pad, top_pad, right_pad, bottom_pad = 10, 10, 10, 10
    print("hihi")
    print(corners.shape)
    
    print(corners.reshape((-1,1,2)).shape)
    rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
    print(f"rec: {rect}")
    box = cv2.boxPoints(rect)
    box_corners = np.int32(box)
    print(f"box corners: {box_corners}")
    #     box_corners = minimum_bounding_rectangle(corners)
    vutt_corners = box_corners.tolist()
    print(vutt_corners)
    epsilon = 1
    for i, cor in enumerate(vutt_corners):
        print(f"cor: {cor}")
        if not(0-epsilon < cor[0] < imW+epsilon):
            # print("wrh")
            out_point = cor
            print((i-1)%len(vutt_corners), (i+1%len(vutt_corners)))
            left_point, right_point = vutt_corners[(i-1)%len(vutt_corners)], vutt_corners[(i+1)%len(vutt_corners)]
            intersec1 = find_intersection(left_point, out_point, [imW, 0], [imW, 3000])
            intersec2 = find_intersection(right_point, out_point, [imW, 0], [imW, 3000])
            vutt_corners.remove(out_point)
            vutt_corners.append(intersec1)
            vutt_corners.append(intersec2)
        
        if not(0-epsilon <= cor[1] < imH+epsilon):
            out_point = cor
            print((i-1)%len(vutt_corners), (i+1%len(vutt_corners)))
            left_point, right_point = vutt_corners[(i-1)%len(vutt_corners)], vutt_corners[(i+1)%len(vutt_corners)]
            intersec1 = find_intersection(left_point, out_point, [0, imH], [3000, imH])
            intersec2 = find_intersection(right_point, out_point, [0, imH], [3000, imH])
            vutt_corners.remove(out_point)
            vutt_corners.append(intersec1)
            vutt_corners.append(intersec2)
            
            


    box_x_min = np.min(box_corners[:, 0])
    box_x_max = np.max(box_corners[:, 0])
    box_y_min = np.min(box_corners[:, 1])
    box_y_max = np.max(box_corners[:, 1])

    # Find corner point which doesn't satify the image constraint
    # and record the amount of shift required to make the box
    # corner satisfy the constraint
    if box_x_min <= 0:
        left_pad = abs(box_x_min) + BUFFER

    if box_x_max >= imW:
        right_pad = (box_x_max - imW) + BUFFER

    if box_y_min <= 0:
        top_pad = abs(box_y_min) + BUFFER

    if box_y_max >= imH:
        bottom_pad = (box_y_max - imH) + BUFFER

    # new image with additional zeros pixels
    image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image.dtype)

    # adjust original image within the new 'image_extended'
    image_extended[top_pad : top_pad + imH, left_pad : left_pad + imW, :] = image
    image_extended = image_extended.astype(np.float32)

    # shifting 'box_corners' the required amount
    box_corners[:, 0] += left_pad
    box_corners[:, 1] += top_pad

    corners = box_corners
    image = image_extended

print(corners.shape)
temp_img = image
corners = sorted(corners.tolist())
corners = order_points(corners)
print(f"vutt: {vutt_corners}")


for point in vutt_corners:
    cv2.circle(temp_img, point, 5, (0, 255, 0), -1)  # -1 for filled circle
    cv2.imwrite("circle.jpg", temp_img)
destination_corners = find_dest(corners)
print(f"des: {destination_corners}")
M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
print(f"des: {destination_corners}")
print(M)
final = cv2.warpPerspective(image, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
final = np.clip(final, a_min=0, a_max=255)
final = final.astype(np.uint8)
cv2.imwrite("output.jpg", final)
