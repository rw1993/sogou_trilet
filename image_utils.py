from PIL import Image
import numpy


def image_to_numpy(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256), Image.ANTIALIAS)
    return numpy.asarray(image, dtype="float")




if __name__ == "__main__":
    image_dir_path = "/media/rw/DATA/sogou/formalCompetition4/News_pic_info_train"
    image_name = image_dir_path + "/{}".format("2016999966.jpg")
    print(image_to_numpy(image_name).shape)
