import layoutparser as lp
import cv2
import re
import os
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader




def get_ordered_layout_list(model, pdf_path, for_math=False):
    # Convert each page of the PDF to a PNG file and save it in the output folder
    pages = convert_from_path(pdf_path) # list type

    # list of list 
    layout_list = []
    image_list = []
    for i, page in enumerate(pages):

        image = np.array(pages[i])

        layout = model.detect(image)
        # re assign the id from left to right, top to bottom

        h, w = image.shape[:2]

        left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

        left_blocks = layout.filter_by(left_interval, center=True)
        left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
        # The b.coordinates[1] corresponds to the y coordinate of the region
        # sort based on that can simulate the top-to-bottom reading order 
        right_blocks = lp.Layout([b for b in layout if b not in left_blocks])
        right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

        # And finally combine the two lists and add the index
        layout = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

        # detect text from title blocks
        ocr_agent = lp.TesseractAgent(languages='eng')
        title_pattern = r"^\d+.*" 

        # use RE to filter real titles
        for b in layout:
            segment_image = (b
                        .pad(left=5, right=5, top=5, bottom=5)
                        .crop_image(image))
            # add padding in each image segment can help
            # improve robustness 
            text = ocr_agent.detect(segment_image)
            b.set(text=text, inplace=True)

            if b.type=='Title':   
                if not re.match(title_pattern, b.text):
                    b.type = 'Text'

        layout_list.append(layout)
        image_list.append(image)

    if for_math:
        return layout_list, image_list
    else:
        # change back the first page first block to title
        layout_list[0][0].type = 'Title'
        return layout_list, image_list
    

#------------------------------------------------------------
# get the image discription
def get_image_discription(page_layout, index_j):
    if 0 <= (index_j + 1) < len(page_layout):
        image_discription = page_layout[index_j + 1].text
        # if the text start with 'Figure' or 'Table', return it
        if re.match(r"Figure|Table", image_discription):
            return image_discription
    if 0 <= (index_j - 1) < len(page_layout):
        image_discription = page_layout[index_j - 1].text
        if re.match(r"Figure|Table", image_discription):
            return image_discription
        
    return ""

#------------------------------------------------------------
def extract_raw_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    raw_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

#------------------------------------------------------------
def has_equation(pdf_path):
    math_ocr_dict = {}

    # only for math part
    model = lp.Detectron2LayoutModel('lp://MFD/faster_rcnn_R_50_FPN_3x/config', 
                                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                    label_map={1: "Equation"})

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
