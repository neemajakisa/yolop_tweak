def csv2cvat(csv_path, img_full_path,output_path):
   
    annotations = []
    images = []
    segmentation = []
    image_id = 0; det_id = 0; iscrowd = 0;
   
    cls = ['ALLIGATOR',  'BLOCK','TRANSVERSE', 'PATCHING', 'SEALING', 'LONGITUDINAL', 'MANHOLE']
    clss = [0,1,2,3,4,5,6]
    categories = []
    for cl in cls:
        cls_id = cls.index(cl)+1
        categories.append({"supercategory": "", "id":cls_id, "name":cl})
   
    img_added = []
    img_added_ = []
    # print (categories)


    df = pd.read_csv(csv_path)
    files = df['image'].unique()
    for curfile in files:
        img_added_.append(curfile)
        cfilepath = os.path.join(img_full_path, curfile)
        print (cfilepath)
        im = cv2.imread(cfilepath)
        img_height = im.shape[0]
        img_width = im.shape[1]
   
        df_cur = df[df['image'] == curfile]
        data_configs = df_cur[['cls','score','x1','y1','x2','y2']].values
        for data_config in data_configs:
         
          w = (data_config[4] - data_config[2])
          h = (data_config[5] - data_config[3])
          area = w*h
        #   print (data_config, area)
   
          bbox = [data_config[2], data_config[3], w, h]
   
          category_id = clss.index(data_config[0]) + 1
          cobj = {'segmentation': segmentation, 'category_id': category_id, 'id': det_id, 'area': area,
            'iscrowd': iscrowd, 'bbox': bbox, 'image_id': image_id}
          annotations.append(cobj)
          if not curfile in img_added:
              cimgs = {"flickr_url": "", "id": image_id, "date_captured": 0, "width": img_width,
              "license": 0, "file_name": curfile.replace('.toml',''),
              "coco_url": "", "height": img_height}
              images.append(cimgs)
              img_added.append(curfile)
   
          det_id += 1
        image_id+=1
   
    df_json = {}
    df_json['info'] = {"contributor": "", "year": "", "description": "", "version": "", "url": "", "date_created": ""}
    df_json['annotations'] = annotations
    df_json['images'] = images
    df_json['categories'] = categories
    df_json['licenses'] = [{"url": "", "id": 0, "name": ""}]
    with open(output_path, 'w') as fp:
        json.dump(df_json, fp)