import os
import copy
import re
import numpy as np
from skimage.draw import rectangle
from skimage.transform import downscale_local_mean
from skimage import img_as_float32
import xml.etree.ElementTree as ET
from aicsimageio import AICSImage
import napari


def get_stitch_info(xml_p):

    # Brief description of structure of matl olympus file
    # < root >
    #   < stage > --> Stage properties
    #   < map >
    #   Series of:
    #   < group > --> Each different stitch area
    #       < regionInfo > --> One per group
    #           < shape >
    #           < coordinates >
    #       < protocolGroupId >
    #       < taskType >
    #       < enable >
    #       < stitching >
    #       < areaInfo > --> Also one per group
    #           < numOfXAreas >
    #           < numOfYAreas >
    #           < areaLeft >
    #           < areaTop >
    #           < areaWidth >
    #           < areaHeight >
    #       Series of:
    #       < area > --> Each subsection of group
    #           < protocolId >
    #           < image >
    #           < xIndex >
    #           < yIndex >
    #   <cycle>

    # Brief overview:
    # Explanation for largest data set acquired so far. Each "group" represents a "stitch", each "area" within the group is a "stack".

    # Enter name spaces information 
    ns_d = {'xsi':'http://www.w3.org/2001/XMLSchema-instance',
            'marker': 'http://www.olympus.co.jp/hpf/model/marker',
            'matl' : 'http://www.olympus.co.jp/hpf/protocol/matl/model/matl'}

    xml_tree = ET.parse(xml_p);xml_root = xml_tree.getroot()
    olymp_d = {
        'stage_rng' : {},
    }
    # Get full range of stage
    stageRng_el = xml_root.find('./matl:stage/matl:strokeRange',ns_d)
    for s_el in stageRng_el: 
        olymp_d['stage_rng'][s_el.tag.split('}')[1]] = int(s_el.text)

    # Get groups data
    groups_l = xml_root.findall('./matl:group',ns_d)
    groups_d = {k:{} for k in range(len(groups_l))}

    # Iterate through group elements and find relevant data
    for grp_idx,e_grp in enumerate(groups_l):
        # Shape of roi-------------
        groups_d[grp_idx]['shape'] = e_grp.find('./marker:regionInfo/marker:shape',ns_d).text
        # Coordinates of roi-------
        grp_crd_d  = e_grp.find('./marker:regionInfo/marker:coordinates',ns_d).attrib
        groups_d[grp_idx]['coordinates'] = {k:int(grp_crd_d[k]) for k in grp_crd_d}
        # Some group properties----
        groups_d[grp_idx]['protocolGroupId'] = e_grp.find('./matl:protocolGroupId',ns_d).text
        groups_d[grp_idx]['enable'] = e_grp.find('./matl:enable',ns_d).text
        groups_d[grp_idx]['stitching'] = e_grp.find('./matl:stitching',ns_d).text
        # Area information (seems to be overlapping with coordinates)
        groups_d[grp_idx]['areaInfo'] = {}
        for e_areaInfo  in e_grp.find('./matl:areaInfo',ns_d):
            groups_d[grp_idx]['areaInfo'][e_areaInfo.tag.split('}')[1]] = int(e_areaInfo.text)
        # Find all areas within this group
        area_l  = e_grp.findall('./matl:area',ns_d)
        groups_d[grp_idx]['areas'] = {k:{} for k in range(len(area_l))}
        # Iterate through all areas in group getting relevant data
        for area_idx,e_area in enumerate(area_l):
            groups_d[grp_idx]['areas'][area_idx]['image'] =  e_area.find('./matl:image',ns_d).text
            groups_d[grp_idx]['areas'][area_idx]['xIndex'] = int(e_area.find('./matl:xIndex',ns_d).text)
            groups_d[grp_idx]['areas'][area_idx]['yIndex'] = int(e_area.find('./matl:yIndex',ns_d).text)

    olymp_d['groups'] = groups_d
    #Get specific cycle data -------------
    olymp_d['cycle'] = {
        'interval' : float(xml_root.find('./matl:cycle/matl:interval',ns_d).text),
        'count' : int(xml_root.find('./matl:cycle/matl:count',ns_d).text)
    }

    return olymp_d

def add_image_info(img_p,info_d):
    '''
    Loads metadata from stitch and add them in place to active groups
    '''
    # Take only enabled groups to load images -------------------------------------------------------------------------------------------------------
    acq_ids = [k for k in info_d['groups'].keys() if info_d['groups'][k]['enable'] == 'true']

    # Follow through parameters to get unified representation
    stage_phys_d = {'y':[],'x':[]}
    img_phys_d = {'y':[],'x':[]}

    for s_id in acq_ids:
        
        # Get remaining info from image for each group ---------------------------------------------------------------------------------------------------------------------
        grp_id = re.search(r'group(\d+)',info_d['groups'][s_id]['protocolGroupId']).group(1)
        fname_0 = [re.search(r'Stitch.*G'+grp_id.zfill(3)+'.oir',s) for s in os.listdir(img_p)] 
        fname = [i.group() for i in fname_0 if not(i is None)]
        
        stitch_info_d = {'pxls_dims':{},'phys_dims':{},'acq_specs':{}}
        try:
            assert len(fname) == 1, 'Warning: multiple names or no names where found with these specs'
            inp_img = AICSImage(os.path.join(img_p,fname[0])) 
            # Populate dictionary with detailes ------------------------------------------------------
            # Dimension and positioning ----------------------------------------
            stitch_info_d['pxls_dims']['x'] = inp_img.dims.X
            stitch_info_d['pxls_dims']['y'] = inp_img.dims.Y
            stitch_info_d['pxls_dims']['z'] = inp_img.dims.Z
            stitch_info_d['phys_dims']['x'] = inp_img.physical_pixel_sizes.X ; img_phys_d['x'].append(inp_img.physical_pixel_sizes.X)
            stitch_info_d['phys_dims']['y'] = inp_img.physical_pixel_sizes.Y; img_phys_d['y'].append(inp_img.physical_pixel_sizes.Y)
            stitch_info_d['phys_dims']['z'] = inp_img.physical_pixel_sizes.Z
            
            # Get stage units to pixel conversion --------------------------------------------------------------------------
            # For reference origin coordinates for groups as defined in 'coordinates' field and 'areaInfo' field match
            # 'x' in coordinates matches 'areaLeft' in areaInfo
            # 'y' in coordinates matches 'areaTop' in areaInfo
            # Assuming a distribution where origin is upper-left corner, x goes positive to right and y goes positive down
            x_phys_range =  inp_img.dims.X*inp_img.physical_pixel_sizes.X
            y_phys_range =  inp_img.dims.Y*inp_img.physical_pixel_sizes.Y
            stage_phys_d['x'].append(np.round(x_phys_range/info_d['groups'][s_id]['coordinates']['width'],6))
            stage_phys_d['y'].append(np.round(y_phys_range/info_d['groups'][s_id]['coordinates']['height'],6))

            assert len(inp_img.metadata.images) == 1, 'Warning: More than in image item found in metadata'
            stitch_info_d['acq_specs']['z_abs'] = [i.position_z for i  in inp_img.metadata.images[0].pixels.planes if i.the_c == 0]
            stitch_info_d['acq_specs']['img_type'] = str(inp_img.metadata.images[0].pixels.type)
            # Channel specs --------------------------------------------------------------
            stitch_info_d['acq_specs']['channels'] = {}
            for e_chn_idx, e_chn in enumerate(inp_img.metadata.images[0].pixels.channels):
                stitch_info_d['acq_specs']['channels'][e_chn_idx] = {
                    'exc':e_chn.excitation_wavelength,
                    'em':e_chn.emission_wavelength,
                }
            # Objective -------------------------------------------------------------------
            stitch_info_d['acq_specs']['objective'] = {
                'mag': inp_img.metadata.instruments[0].objectives[0].nominal_magnification,
                'model':inp_img.metadata.instruments[0].objectives[0].model,
            }

            info_d['groups'][s_id]['imgSpecs'] = copy.deepcopy(stitch_info_d)

            


        except AssertionError as mssg:
            print(mssg)
            info_d['groups'][s_id]['imgSpecs'] = None
    
    # Stage dimensions -----------------------------------------
    try:
        assert (np.all(np.array(stage_phys_d['x'])==stage_phys_d['x'][0]) and\
        np.all(np.array(stage_phys_d['y'])==stage_phys_d['y'][0]) and\
        stage_phys_d['x'][0]==stage_phys_d['y'][0]), 'Warning, stage dimensions not conistent or units not square'
        info_d['stage_phys_dims'] =  stage_phys_d['x'][0]             
    except AssertionError as mssg:
        print(mssg)        

    # Image pixel size -----------------------------------------
    try:
        assert(np.all(np.array(img_phys_d['x'])==img_phys_d['x'][0]) and\
               np.all(np.array(img_phys_d['y'])==img_phys_d['y'][0]) and\
               img_phys_d['x'][0] == img_phys_d['y'][0]), 'Warning, pixel dimensions not conistent or not square'
        info_d['pxl_phys_dims'] = img_phys_d['x'][0] 
    except AssertionError as mssg:
        print(mssg)



# Function to return image file -----------------------------------------------------------------
def load_img(img_p,info_d,item_id):
    
    group_id = re.search(r'group(\d+)',info_d['groups'][item_id]['protocolGroupId']).group(1)
    fileName_0 = [re.search(r'Stitch.*G'+group_id.zfill(3)+'.oir',s) for s in os.listdir(img_p)] 
    fileName = [i.group() for i in fileName_0 if not(i is None)]
    inputImg_0 = AICSImage(os.path.join(img_p,fileName[0])) 
    inputImg = inputImg_0.data

    return inputImg
# ---------------------------------------------------------------------------------------------------- 

# ---------------------------------------------------------------------------------------------------- 
def get_map_imgs_orig(map_p,map_info,map_ids,map_ant_to_post_ids,track_bounds_d,stg_to_um,down_sample=1):

    mapProj_pxl_size = map_info['pxl_phys_dims']*down_sample
    # Obtain used stage pixel extent 
    x_pxl_ext = int(np.ceil(track_bounds_d['width']/mapProj_pxl_size))
    v_pxl_ext = int(np.ceil(track_bounds_d['height']/mapProj_pxl_size))
    mapProj_msk = np.zeros([v_pxl_ext,x_pxl_ext])

    try:
        # Check if all slides are accounted for ----
        assert np.array_equal(np.sort(map_ant_to_post_ids),np.array(map_ids)), 'WARNING: Not all slides are there' 
        map_imgs = []
        for s_map in map_ant_to_post_ids:

            # Load image and downscale ----------------------------------------------------------------
            s_map_img_0 = load_img(map_p,map_info,s_map)
            s_map_img_float = img_as_float32(s_map_img_0)
            s_map_img = downscale_local_mean(s_map_img_float[0,0,0,...],(down_sample,down_sample))
            # Calculate image offset in pixels -------------------------------------------------------
            s_map_v0 = (map_info['groups'][s_map]['coordinates']['y']*stg_to_um-track_bounds_d['min_v'])/mapProj_pxl_size
            s_map_x0 = (map_info['groups'][s_map]['coordinates']['x']*stg_to_um-track_bounds_d['min_x'])/mapProj_pxl_size
            s_map_crds = rectangle((s_map_v0,s_map_x0),extent=s_map_img.shape)

            mapProj_msk[np.ravel(s_map_crds[0],order='F'),np.ravel(s_map_crds[1],order='F')] = np.ravel(s_map_img,order='C')
            map_imgs.append(s_map_img)

        map_imgs = np.array(map_imgs)


    except AssertionError as mssg:
        print(mssg)
        map_imgs = None

    return map_imgs,mapProj_msk,mapProj_pxl_size


def get_stack_imgs_orig(stack_p,stack_info,stack_ids,ant_to_post_ids,mode = 'single'):

    # mode argument can be one of 'single' or 'full' to select only loading max-proj of first channel or all data set respectively

    # Load registration channels for all images ----------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------- 
    # For pooling all data together it is important to note that acquisition order does not follow sequential order in brain
    # Check if all slides are accounted for ----
    try:
        assert np.array_equal(np.sort(ant_to_post_ids),np.array(stack_ids)),'WARNING: Not all slides are there'
        reg_imgs_l = []

        for e_s_stack in ant_to_post_ids:    
            img_0 = load_img(stack_p,stack_info,e_s_stack)
            if mode == 'single':
                img_0_t = img_as_float32(img_0[0,0,...])
                img_0_to_load = np.mean(img_0_t,axis=0)
            elif mode == 'full':
                img_0_to_load = img_as_float32(img_0[0,...])
            reg_imgs_l.append(img_0_to_load)
        reg_imgs = np.array(reg_imgs_l)  

    except AssertionError as mssg:
        print(mssg) 
        reg_imgs = None

    return reg_imgs


# ---------------------------------------------------------------------------------------------------- 

def adjust_zoomed(command,ch_ids,n_lyout,slices_ids, spec_arg = None):
    '''
    command: either hide/show/contrast/gamma followed by '_' and either all/number of slice
    if command == 'gamma' spec value with spec_arg
    if command == 'constrast' spec percentiles bounds as two element list in spec_arg    
    if command == 'colormap' spec colormap as spec_arg
    '''


    if command.split('_')[-1] == 'all':
        for e_s_slc in slices_ids:
            for e_s_ch in ch_ids:
                if command.split('_')[0] == 'hide':
                    n_lyout.layers[f'slc_{e_s_slc}_{e_s_ch}'].visible = False
                elif command.split('_')[0] == 'show':
                    n_lyout.layers[f'slc_{e_s_slc}_{e_s_ch}'].visible = True
                elif command.split('_')[0] == 'gamma':
                    n_lyout.layers[f'slc_{e_s_slc}_{e_s_ch}'].gamma = spec_arg
                elif command.split('_')[0] == 'contrast':
                    p_0 = np.percentile(n_lyout.layers[f'slc_{e_s_slc}_{e_s_ch}'].data,spec_arg[0])
                    p_1 = np.percentile(n_lyout.layers[f'slc_{e_s_slc}_{e_s_ch}'].data,spec_arg[1])
                    n_lyout.layers[f'slc_{e_s_slc}_{e_s_ch}'].contrast_limits = (p_0,p_1)
                elif command.split('_')[0] == 'colormap':
                    n_lyout.layers[f'slc_{e_s_slc}_{e_s_ch}'].colormap = spec_arg
    else:           

        slc_id = int(command.split('_')[-1])
        try:
            assert slc_id in slices_ids, 'WARNING: provided slice is not within bounds'
            for e_s_ch in ch_ids:
                if command.split('_')[0] == 'hide':
                    n_lyout.layers[f'slc_{slc_id}_{e_s_ch}'].visible = False
                elif command.split('_')[0] == 'show':
                    n_lyout.layers[f'slc_{slc_id}_{e_s_ch}'].visible = True
                elif command.split('_')[0] == 'gamma':
                    n_lyout.layers[f'slc_{slc_id}_{e_s_ch}'].gamma = spec_arg
                elif command.split('_')[0] == 'contrast': 
                    p_0 = np.percentile(n_lyout.layers[f'slc_{slc_id}_{e_s_ch}'].data,spec_arg[0])
                    p_1 = np.percentile(n_lyout.layers[f'slc_{slc_id}_{e_s_ch}'].data,spec_arg[1])
                    n_lyout.layers[f'slc_{slc_id}_{e_s_ch}'].contrast_limits = (p_0,p_1)
                elif command.split('_')[0] == 'colormap':
                    n_lyout.layers[f'slc_{slc_id}_{e_s_ch}'].colormap = spec_arg
        except AssertionError as mssg:
            print(mssg)

    return n_lyout


def visualize_layout(whole_proj,stack_inp,map_proj_pxl_size,stack_info,ant_to_post_ids,stg_to_um,track_bounds_d,mode='dual'):

    n_vwr_7 = napari.Viewer()
    n_vwr_7.add_image(whole_proj,scale = [map_proj_pxl_size,map_proj_pxl_size])


    if mode == 'dual':
        specs_cmap = ['Green','Red']
        chnl_names = ['green','red']
        inp_imgs = stack_inp[:,1:,...]
    elif mode == 'full':
        specs_cmap = ['bop blue','Green','Red']  
        chnl_names = ['blue','green','red']
        inp_imgs =   stack_inp

    for s_stack_id,s_stack_image in zip(ant_to_post_ids,inp_imgs):        
        stck_x0 = (stack_info['groups'][s_stack_id]['coordinates']['x']* stg_to_um-track_bounds_d['min_x'])
        stck_v0 = (stack_info['groups'][s_stack_id]['coordinates']['y']* stg_to_um-track_bounds_d['min_v'])
        stck_zProj = np.max(s_stack_image,axis=1) 
        chnl_names_iter = [f'slc_{s_stack_id}_{s}' for s in chnl_names]
        n_vwr_7.add_image(stck_zProj,channel_axis=0,name=chnl_names_iter,colormap= specs_cmap,scale = [stack_info['pxl_phys_dims'],stack_info['pxl_phys_dims']],\
                        translate = [stck_v0,stck_x0])

    return n_vwr_7
