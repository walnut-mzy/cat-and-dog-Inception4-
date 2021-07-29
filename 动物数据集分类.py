#C:\Users\mzy\Desktop\机器学习\data\train
import tensorflow as tf
import random
import os
def image_deals1(train_file):       # 读取原始文件
    image_string = tf.io.read_file(train_file)  # 读取原始文件
    image_decoded = tf.image.decode_png(image_string)  # 解码JPEG图片
    image_decoded=randoc(image_decoded)
    image_decoded= tf.image.resize(image_decoded, [299, 299])  #把图片转换为224*224的大小
    #image = tf.image.rgb_to_grayscale(image_decoded)
    image = tf.cast(image_decoded, dtype=tf.float32) / 255.0-0.5
    return image
def image_deals(train_file):       # 读取原始文件
    image_string = tf.io.read_file(train_file)  # 读取原始文件
    image_decoded = tf.image.decode_png(image_string)  # 解码JPEG图片
    image_decoded=randoc(image_decoded)
    image_decoded= tf.image.resize(image_decoded, [299, 299])  #把图片转换为224*224的大小
    #image = tf.image.rgb_to_grayscale(image_decoded)
    image = tf.cast(image_decoded, dtype=tf.float32) / 255.0-0.5
    return image
def randoc(train_file):
    int1=random.randint(1,10)
    if int1==1:
        train_file = tf.image.random_flip_left_right(train_file)   #左右翻折
    elif int1==2:
        train_file=tf.image.random_flip_up_down(train_file)
    return train_file

def train_test_get(train_test_inf):
    for root,dir,files in os.walk(train_test_inf, topdown=False):
        #print(root)
        #print(files)
        list=[root+"/"+i for i in files]
        #print(list)
        filename=[]
        for i in files:
            label=i[0:3]
            if label=="cat":
                #x1 = tf.constant([0, 1], shape=(1, 2))
                x1=[0,1]
                filename.append(x1)
            else:
                #x2 = tf.constant([1, 0], shape=(1, 2))
                x2=[0,1]
                filename.append(x2)

        json={
            "list":list,
            "filename":filename
        }
        print(len(list))
        print(len(filename))
        return json
def dogandcat():
    json_train=train_test_get("C:/Users/mzy/Desktop/机器学习/data/train1")
    list_file=json_train["list"]
    list_filename=json_train["filename"]
    print(list_file)
    image_list=[image_deals(i) for i in list_file]
    #image_list=tf.expand_dims(image_list,axis=1)
    # print(image_list.shape)
    dataest=tf.data.Dataset.from_tensor_slices((image_list, list_filename))
    dataest=dataest.shuffle(buffer_size=300).repeat(count=10).prefetch(tf.data.experimental.AUTOTUNE).batch(10)

    print(dataest)
    return dataest
#dogandcat()
def dogandcat1():
    json_train=train_test_get("C:/Users/mzy/Desktop/机器学习/data/test1")
    list_file=json_train["list"]
    list_filename=json_train["filename"]
    print(list_file)
    image_list=[image_deals(i) for i in list_file]
    #image_list=tf.expand_dims(image_list,axis=1)
    # print(image_list.shape)
    dataest=tf.data.Dataset.from_tensor_slices((image_list, list_filename))
    dataest=dataest.shuffle(buffer_size=300).repeat(count=10).prefetch(tf.data.experimental.AUTOTUNE).batch(10)

    #print(dataest)
    return dataest