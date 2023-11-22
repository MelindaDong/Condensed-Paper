from functions import get_ordered_layout_list
import gpt_function
import layoutparser as lp
from PIL import Image


def has_equation(pdf_path):
    math_ocr_dict = {}

    # only for math part
    model = lp.Detectron2LayoutModel('lp://MFD/faster_rcnn_R_50_FPN_3x/config', 
                                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                    label_map={1: "Equation"})
    
    # # # has math, but no title, overall not accurate image
    # model = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config', 
    #                                  extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
    #                                  label_map={1: "TextRegion", 2: "ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"})

    layout_list, image_list = get_ordered_layout_list(model, pdf_path, for_math=True)

    has_equation = False
    # save all the math images
    for i, layout in enumerate(layout_list):
        for j, b in enumerate(layout):
            if b.type=='Equation':
                image = (b
                        .pad(left=30, right=30, top=15, bottom=15)
                        .crop_image(image_list[i]))
                image = Image.fromarray(image)
                path_name = f'math_images/{i}_{j}.png'
                image.save(path_name)
                # save the path to math_ocr_dict
                math_ocr_dict[path_name] = b.text
                has_equation = True
    
    print("get all math images")
    # save math_ocr_dict
    math_ocr_text = repr(math_ocr_dict)
    with open("math_ocr_text.txt", "w") as file:
        file.write(math_ocr_text)


    return has_equation, math_ocr_dict


# test the function
pdf_path = 'test.pdf'
has_equation, math_dict = has_equation(pdf_path)
print("finish running get_math.py")