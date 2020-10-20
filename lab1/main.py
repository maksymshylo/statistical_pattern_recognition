from functionality import *

def main():

    '''
    Parameters
        argv[1]: str 
            path to json
        argv[2]: str
            path to image folder
        argv[3]: string
            input string
        argv[4]: float
            noise level
    '''
    # initialise parameters
    path_to_json, image_folder, input_string, p = sys.argv[1:]
    a = time.time()
    p = float(p)
    alphabet_list, alphabet_dict, p_k =  get_bigrams(path_to_json)
    reference_images = import_images(image_folder,alphabet_list)
    input_image_zero_noise = string_to_image(input_string,reference_images,0)
    input_image = string_to_image(input_string,reference_images,p)

    # recognizing string 
    output_image = recognizer(input_image,alphabet_list,reference_images,p,p_k)
    print("total time", time.time()-a)
    print("length:",len(input_string))
    print("input string : ", input_string)

    # plot images
    plt.figure()

    plt.subplot(3,1,1), plt.imshow(input_image_zero_noise,cmap = 'gray'), plt.axis('off')

    plt.subplot(3,1,2), plt.imshow(input_image,cmap = 'gray'), plt.axis('off')

    plt.subplot(3,1,3), plt.imshow(output_image,cmap = 'gray'), plt.axis('off')

    plt.show()    


if __name__ == "__main__":
    main()
