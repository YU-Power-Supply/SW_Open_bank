#--V 1.1--
import sys
import os
import json

def create_dir(dname):
    try:
        if not os.path.exists("./"+dname): # if you don't have same dir, create
            os.makedirs("./"+dname)
    except OSError:
        print ('Error: Creating directory. ' +  dname)

def make_data(dname, classes): #make obj.data file
    with open(f'{dname}/obj.data', "w", encoding='UTF8') as f:
        strdata = ''
        data = {'classes = ' : '{}\n'.format(len(classes)), #count classes
        'train = ' : f'{dname}train.txt\n', # train data path file
        'valid = ' : f'{dname}train.txt\n', # valid data path file
        'names = ' : f'{dname}obj.names\n', # classes name file
        'backup = ' : 'backup/\n'}           # storage path

        for k, v in data.items():
            strdata += k + v
        f.write(strdata)

def make_names(dname, classes): # make obj.names file

    print("change [region]        classes = ", len(classes))
    print("change [convolutional] filters = ", (len(classes)+5)*5)
    with open(f'{dname}/obj.names', "w", encoding='UTF8') as f:
        f.write('\n'.join(classes))

def make_train_list(dname, img_dir_list): # make train.txt file
    file_list = []
    for path in img_dir_list:
        file_list += [path+file for file in os.listdir(path) if file[-1] == 'g'] # collect png, jpg etc..
    print(file_list)
    with open(f'{dname}/train.txt', "w", encoding='UTF8') as f:
        f.write('\n'.join(file_list))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## 디렉토리 이름 변경 ##
def rename_dir(img_dir):
    dir_list = []
    for dir_name in os.listdir(img_dir):
        if 7 < len(dir_name): # if longer then 6, rename
            os.rename(img_dir+dir_name, img_dir+dir_name[:6])
        dir_list.append(img_dir+dir_name[:6] + "/")
    return dir_list

## 한 번에 합쳐보리기 ##
def rename_file(dir_list):
    for file_name in dir_list:
        pass # 보류

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def json_to_txt(json_path, save_path, classes):
    file_list = os.listdir(json_path)
    
    for i in file_list:
        data = ''
        with open(json_path+i, encoding='UTF8') as file:
            json_data = json.load(file)
            bounding_data = json_data['ObjectInfo']['BoundingBox']
            for obj in bounding_data:
                if bounding_data[obj]['isVisible']:
                    ltx, lty, rbx, rby = map(float, bounding_data[obj]['Position'])
                    #conversion x, y, w, h
                    transpos = '{} {} {} {}'.format(((ltx + rbx)/2)/720, ((lty + rby)/2)/1280, (rbx - ltx)/720, (rby - lty)/1280)
                    data += f"{classes[obj]} {transpos}\n"

        #데이터 저장부
        savepath = save_path + i[:-4] + "txt" # json -> txt
        with open(savepath, "w", encoding='UTF8') as wfile:
            wfile.write(data)
    
    
def remove_txt(txtpath):  # when you want to remove txt files
    for t in os.listdir(txtpath):
        if t[-3:] == "txt":
            os.remove(txtpath+t)

if __name__ == '__main__':
    
    directory = 'custom/' # what you want to create directory name
    img_dir = 'Training/customboximg/' #route of img_dir
    json_dir = 'Training/custombox/'    #route of json_dir
    classes = {'Face' : 0, 'Leye' : 1, 'Reye' : 2, 'Mouth' : 3, 'Phone' : 4, 'Cigar' : 5}
    
    img_dir_list = rename_dir(img_dir)
    json_dir_list = rename_dir(json_dir)

    img_dir_list.sort()
    json_dir_list.sort()

    if len(sys.argv) == 1:
        print("명령 프롬프트로 실행하세요")
        exit(0)
    elif sys.argv[1] == "set":
        create_dir(directory)
        make_data(directory, classes)
        make_names(directory, classes)
        make_train_list(directory, img_dir_list)
    elif sys.argv[1] == "jtt":
        for i in range(len(json_dir_list)):
            print(i)
            json_to_txt(json_dir_list[i], img_dir_list[i], classes)
        print("완료")
    elif sys.argv[1] == "rmv":
        for i in img_dir_list:
            print(i)
            remove_txt(i)
