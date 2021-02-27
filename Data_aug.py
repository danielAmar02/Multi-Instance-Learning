import numpy as np
import random
import threading
#from keras.preprocessing.image import ImageDataGenerator
import scipy.misc as sci
import cv2

def random_flip_img(img, horizontal_chance=0, vertical_chance=0):
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    if not isinstance(img, list):
        res = cv2.flip(img, flip_val) # 0 = X axis, 1 = Y axis,  -1 = both
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
    return res

def random_rotate_img(images):
    rand_roat = np.random.randint(4, size=1)
    angle = 90*rand_roat
    center = (images.shape[0] / 2, images.shape[1] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle[0], scale=1.0)

    img_inst = cv2.warpAffine(images, rot_matrix, dsize=images.shape[:2], borderMode=cv2.BORDER_CONSTANT)

    return img_inst

def random_crop(image, crop_size=(400, 400)):
    height, width = image.shape[:-1]
    dy, dx = crop_size
    X = np.copy(image)
    aX = np.zeros(tuple([3, 400, 400]))
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    aX = X[y:(y + dy), x:(x + dx), :]
    return aX

class threadsafe_iter(object):
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

class DataGenerator(object):
    def __init__(self, batch_size=32, shuffle=True):
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __Get_exploration_order(self, list_patient, shuffle):
        indexes = np.arange(len(list_patient))
        if shuffle:
            random.shuffle(indexes)
        return indexes

    def __Data_Genaration(self, batch_train):
        bag_batch = []
        bag_label = []
        
        #datagen = ImageDataGenerator()
                                                                       
        #transforms = {
        	#	"theta" : 0.25,
        	#	"tx" : 0.2,
        	#	"ty" : 0.2,
        	#	"shear" : 0.2,
        	#	"zx" : 0.2,
        	#	"zy" : 0.2,
        #   "flip_horizontal" : True,
        #		"zx" : 0.2,
        	#	"zy" : 0.2,
        	#}
        
        for ibatch, batch in enumerate(batch_train):
            aug_batch = []
            img_data = batch[0]
            for i in range(img_data.shape[0]):
                ori_img = img_data[i, :, :, :]
                # sci.imshow(ori_img)
                if self.shuffle:
                    img = random_flip_img(ori_img, horizontal_chance=0.5, vertical_chance=0.5)
                    img = random_rotate_img(img)
                    #img = datagen.apply_transform(ori_img,transforms)
                else:
                    img = ori_img
                exp_img = np.expand_dims(img, 0)
                # sci.imshow(img)
                aug_batch.append(exp_img)
            input_batch = np.concatenate(aug_batch)
            bag_batch.append((input_batch))
            bag_label.append(batch[1])

        return bag_batch, bag_label


    def generate(self, train_set):
        flag_train = self.shuffle

        while 1:

        # status_list = np.zeros(batch_size)
        # status_list = []
            indexes = self.__Get_exploration_order(train_set, shuffle=flag_train)

        # Generate batches
            imax = int(len(indexes) / self.batch_size)

            for i in range(imax):
                Batch_train_set = [train_set[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]

                X, y = self.__Data_Genaration(Batch_train_set)

                yield X, y
