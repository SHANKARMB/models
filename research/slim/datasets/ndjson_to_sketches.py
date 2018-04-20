from PIL import Image
from jsonlines import jsonlines
import os
import urllib
import random
import requests


def create_image(image, filename):
    img = Image.new('RGB', (256, 256), "white")
    pixels = img.load()

    x = -1
    y = -1

    for stroke in image:
        for i in range(len(stroke[0])):
            if x != -1:
                for point in get_line(stroke[0][i], stroke[1][i], x, y):
                    pixels[point[0], point[1]] = (0, 0, 0)
            pixels[stroke[0][i], stroke[1][i]] = (0, 0, 0)
            x = stroke[0][i]
            y = stroke[1][i]
        x = -1
        y = -1
    img.save(filename)


def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


rootDir = '/content/training/dataset'  # root dir
# rootDir = '/home/prime/ProjectWork/training/dataset'
current_dataset = os.path.join(rootDir, 'ndjson')  # path where ndjson files exists relative to root dir
final_dir = os.path.join(rootDir, 'cnn_images')  # final path where images are store relative to root dir

os.chdir(rootDir)

# ______________________________________________________________________---------------------------------------____________________________


root_path = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified'

# just names
ndjson_list = ['airplane', 'hot air balloon', 'snake', 'pineapple', 'butterfly',
               'knife', 'wine bottle', 'apple', 'hamburger', 'scissors']

for i in ndjson_list:
    file_url = os.path.join(root_path, urllib.parse.quote(i) + '.ndjson')
    print('downloading ', i, ' -> ', file_url)
    r = requests.get(file_url)
    file_name = i + '.ndjson'
    with open(file_name, mode='wb') as f:
        f.write(r.content)

# _______________________________________________________________________--------------------------------------_____________________________


for ndj_file in os.listdir(current_dataset):
    folder = os.path.join(final_dir, os.path.splitext(os.path.basename(ndj_file))[0])
    os.makedirs(folder, exist_ok=True)
    file_name = 0
    count = 0
    x = random.sample(range(100000), 11000)
    with jsonlines.open(os.path.join(current_dataset, ndj_file), mode='r') as reader:
        for obj in reader:
            if count in x:
                create_image(obj['drawing'], os.path.join(folder, str(file_name) + '.jpg'))
                file_name = file_name + 1
            count = count + 1
    print(ndj_file, 'done')
