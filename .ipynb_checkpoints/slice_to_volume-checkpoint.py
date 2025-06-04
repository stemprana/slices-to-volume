
import os
import numpy as np
import json
import cv2
from scipy import ndimage
from skimage.draw import rectangle, polygon, polygon2mask, rectangle_perimeter
from skimage.transform import downscale_local_mean
from skimage import exposure, measure
import matplotlib.pyplot as plt
import napari
from tifffile import memmap

def load_json_info(json_fname,save_p):

    '''
    Load images specifications saved as json files
    INPUTs: json_fname : string, name of json file with specifications
            save_p : PosixPath, path to folder containing files with specs
    OUTPUTs: inf_out : dict, image file specifications
    '''

    with open(save_p/json_fname,'r') as json_file_in:    
        inf_inp = json.load(json_file_in)    
    # Keys that are going to remain unchanged from json input file
    clone_keys = ['stage_rng','stage_phys_dims','pxl_phys_dims']
    inf_out = {k:inf_inp[k] for k in clone_keys}    
    # Casting 'group' keys from string to integers 
    inf_out['groups'] = {int(k):inf_inp['groups'][k] for k in inf_inp['groups'].keys()}

    return inf_out

def map_global_layout(map_info,stack_info):
    '''
    Display layout of map (zoomed out) and stacks (zoomed in) within full microscope stage range
    INPUTs:     map_info : dict, image file specifications for zoomed-out images
                stack_info : dict, image file specifications for zoomed-in images
    OUTPUTs:    stg_to_um : float, size of stage units in microns 
                track_bounds_d : dict, boundaries in microns defined by images taken within full microscope stage
                map_ids : list, indexes of acquired zoomed-out images 
                stack_ids : list, indexes of acquired zoomed-in images
    '''
    
    try:
        
        # Get stage coordinates in microns ---------------------------------------------------------------------- 
        # Get stage dimension and make sure it is consistent in both maps and stacks 
        assert stack_info['stage_phys_dims'] == map_info['stage_phys_dims'], 'Stage dimensions inconsistencies for zoomed-in and zoomed-out images'
        stg_to_um = map_info['stage_phys_dims']

        v0 = map_info['stage_rng']['minY']*stg_to_um
        v1 = map_info['stage_rng']['maxY']*stg_to_um
        x0 = map_info['stage_rng']['minX']*stg_to_um
        x1 = map_info['stage_rng']['maxX']*stg_to_um
        
        f_0 = plt.figure(layout='constrained',figsize=(12,8),dpi=90)
        a_0 = f_0.add_axes([0.05,0,0.9,0.9])
        a_0.set_title('Global layout visualization',fontname='sans-serif',fontsize=22,ha='center',pad=1,color = 'darkslategrey')

        # Plot stage layout --------------------------------------------------------------------------------------
        stage_ptch = plt.Rectangle([x0,v0],x1-x0,v1-v0,edgecolor = 'darkred',facecolor='None',linestyle='--',label='microscope stage')
        a_0.add_patch(stage_ptch)
        a_0.set_xlim([x0-0.05*(x1-x0),x1+0.05*(x1-x0)])
        a_0.set_ylim([v1+0.05*(v1-v0),v0-0.05*(v1-v0)])
        
        
        track_bounds_d = {'min_x':None,'max_x':None,'min_v':None,'max_v':None}
        # Plot map layouts ---------------------------------------------------------------------------------------
        map_ids = [m for m in map_info['groups'].keys() if map_info['groups'][m]['enable'] == 'true']
        for e_map in map_ids:
            # Get coordinates in microns ---------------------------------------
            m_x0 = map_info['groups'][e_map]['coordinates']['x'] * stg_to_um
            m_v0 = map_info['groups'][e_map]['coordinates']['y'] * stg_to_um
            m_w = map_info['groups'][e_map]['coordinates']['width'] * stg_to_um
            m_h = map_info['groups'][e_map]['coordinates']['height'] * stg_to_um    
            map_ptch = plt.Rectangle([m_x0,m_v0],m_w,m_h,edgecolor = 'steelblue',facecolor='None',linestyle='--', label = 'low res')
            a_0.add_patch(map_ptch)
            a_0.text(m_x0+m_w/2,m_v0+m_h/2,'map_'+str(e_map).zfill(2),color='steelblue',\
                    fontname = 'sans-serif',fontsize=8,fontweight='bold',ha='center')
            # Update bounds ----------------------------------------------------------
            if (track_bounds_d['min_x'] is None) or (m_x0 < track_bounds_d['min_x']): track_bounds_d['min_x'] = m_x0 
            if (track_bounds_d['max_x'] is None) or ( m_x0 + m_w > track_bounds_d['max_x']): track_bounds_d['max_x'] =  m_x0 + m_w
            if (track_bounds_d['min_v'] is None) or ( m_v0 < track_bounds_d['min_v']): track_bounds_d['min_v'] =  m_v0
            if (track_bounds_d['max_v'] is None) or ( m_v0 + m_h > track_bounds_d['max_v']): track_bounds_d['max_v'] = m_v0 + m_h 
       
        # Plot stage layouts -----------------------------------------------------------------------------------
        stack_ids = [s for s in stack_info['groups'].keys() if stack_info['groups'][s]['enable']=='true']
        for e_stack in stack_ids:
            # Get coordinates in microns -----------------------------------------
            s_x0 = stack_info['groups'][e_stack]['coordinates']['x']* stg_to_um
            s_v0 = stack_info['groups'][e_stack]['coordinates']['y']* stg_to_um
            s_w = stack_info['groups'][e_stack]['coordinates']['width']* stg_to_um
            s_h = stack_info['groups'][e_stack]['coordinates']['height']* stg_to_um
            stack_ptch = plt.Rectangle([s_x0,s_v0],s_w,s_h,edgecolor = 'darkolivegreen',facecolor='None',linestyle='--',label = 'high res')
            a_0.add_patch(stack_ptch)
            a_0.text(s_x0+s_w/2,s_v0+s_h/2,'stack_'+str(e_stack).zfill(2),color='darkolivegreen',\
                    fontname = 'sans-serif',fontsize=6,fontweight='bold',ha='center')
            # Update bounds ----------------------------------------------------------
            if ( s_x0 <  track_bounds_d['min_x']): track_bounds_d['min_x'] = s_x0 
            if ( s_x0 + s_w > track_bounds_d['max_x']): track_bounds_d['max_x'] =  s_x0 + s_w
            if  ( s_v0 < track_bounds_d['min_v']): track_bounds_d['min_v'] =  s_v0
            if  ( s_v0 + s_h > track_bounds_d['max_v']): track_bounds_d['max_v'] =  s_v0 + s_h

        # Plot bounds layout -----------------------------------------------------------------
        track_bounds_d['width'] = track_bounds_d['max_x']-track_bounds_d['min_x']
        track_bounds_d['height'] = track_bounds_d['max_v']-track_bounds_d['min_v']

        # Stage span is too big to represent in image. Rescale to optimal stage-to-pixel units by scannign through all stitches
        bounds_ptch = plt.Rectangle([track_bounds_d['min_x'],track_bounds_d['min_v']],track_bounds_d['width'],track_bounds_d['height'],\
                                    edgecolor = 'dimgrey',facecolor='None',linestyle='--',label = 'images bound')
        a_0.add_patch(bounds_ptch)    

        # Add scalebar, with reference to stage lower right corner ---------------------------------------------
        sbar_l = 10 # in mm
        sbar_vh_pad = [2,2] # in mm
        sbar_y = v1 - sbar_vh_pad[0]*1e3
        sbar_1 = x1 -sbar_vh_pad[1]*1e3
        sbar_0 = sbar_1 - sbar_l*1e3
        a_0.hlines(sbar_y,sbar_0,sbar_1,color='darkslategrey',linewidth = 3)
        a_0.text((sbar_0+sbar_1)/2,sbar_y - 500,str(sbar_l)+' mm',fontname='sans-serif',fontsize=15,\
        fontweight = 'normal',ha='center',va='bottom',color='darkslategrey')        
        a_0.legend(handles=[stage_ptch,bounds_ptch,map_ptch,stack_ptch],loc='upper right',shadow=True,prop={'family':'sans-serif','size':10})
        a_0.set_axis_off()
        
    except AssertionError as mssg:
        print(mssg)
        stg_to_um = None
        track_bounds_d = None
        map_ids = None

    return stg_to_um,track_bounds_d,map_ids,stack_ids

def plot_map_imgs(mapProj,stack_info,stack_ids,mapProj_pxl_size,track_bounds_d,stg_to_um,add_zoomed_in = True):
    '''
    Display assembly of low resolution slices images with optional overlay of zoomed in areas
    INPUTs:     mapProj:            array, image displaying assembly of zoomed-out images in stage                
                stack_info:         dict, image file specifications for zoomed-in images
                stack_ids:          list, indexes of acquired zoomed-in images
                mapProj_pxl_size:   float, pixel size in microns
                track_bounds_d:     dict, boundaries in microns defined by images taken within full microscope stage
                stg_to_um:          float, size of stage units in microns 
                add_zoomed_in (optional) : bool, whether or not to display layout of zoomed-in areas (default = True)
    '''

    f_1 = plt.figure(layout='constrained',figsize=(12,8),dpi=90)
    a_1 = f_1.add_axes([0.05,0,0.9,0.9])
    a_1.set_title('Assembly of slices',fontname='sans-serif',fontsize=22,ha='center',pad=1,color = 'darkslategrey')
    a_1.imshow(mapProj,cmap='gray_r')
    a_1.set_axis_off()
    # Add scalebar, with reference to stage lower right corner ---------------------------------------------
    v1 = mapProj.shape[0]
    x1 = mapProj.shape[1]
    sbar_l = 5 # in mm
    sbar_vh_pad = [1,15] # in mm
    sbar_l_pxl = round(sbar_l*1000/mapProj_pxl_size)
    sbar_vh_pad_pxl = np.round(np.array(sbar_vh_pad)*1000/mapProj_pxl_size)
    sbar_y = v1 - sbar_vh_pad_pxl[0]
    sbar_1 = x1 - sbar_vh_pad_pxl[1]
    sbar_0 = sbar_1 - sbar_l_pxl
    a_1.hlines(sbar_y,sbar_0,sbar_1,color='darkslategrey',linewidth = 3)
    a_1.text((sbar_0+sbar_1)/2,sbar_y - 10,str(sbar_l)+' mm',fontname='sans-serif',fontsize=15,\
    fontweight = 'normal',ha='center',va='bottom',color='darkslategrey')        

    
    if add_zoomed_in:
        # Add stack layouts ------------------------------------------------------------------------------------------------------------
        for s_stack in stack_ids:
            # Get coordinates in microns -----------------------------------------------------------------------------------------------
            stck_x0 = (stack_info['groups'][s_stack]['coordinates']['x']* stg_to_um-track_bounds_d['min_x'])/mapProj_pxl_size
            stck_v0 = (stack_info['groups'][s_stack]['coordinates']['y']* stg_to_um-track_bounds_d['min_v'])/mapProj_pxl_size
            stck_w = stack_info['groups'][s_stack]['coordinates']['width']* stg_to_um/mapProj_pxl_size
            stck_h = stack_info['groups'][s_stack]['coordinates']['height']* stg_to_um/mapProj_pxl_size
            mapProj_stack_ptch = plt.Rectangle([stck_x0,stck_v0],stck_w,stck_h,edgecolor = 'darkorange',facecolor='None',linestyle='--',label = 'high res layout')
            a_1.add_patch(mapProj_stack_ptch)
        # ----------------------------------------------------------------------------------------------------------------------------- 
        a_1.legend(handles=[mapProj_stack_ptch],loc='lower right',shadow=True,prop={'family':'sans-serif','size':10})

def detect_flip(stck_imgs, flip_dwn_factor = 10):

    '''
    Algorithm for the detection of brain slice flipping along long axis due to backward mounting
    INPUTs:     stck_imgs:          array, stack of zoomed-in high-resolution images containing channel with ubiquitous labeling of region of interest. 
                                    Images must be sorted from anterior to posterior location. [n_images,y,x]
                flip_dwn_factor:    int, factor by which to downsample images before assesment
    OUTPUTs:    flp_trk:            list, result of flipping assesment relative to first slice. 1 means aligned and -1 reversed. 
                                    First element correspond to first slice and defined as 1. 
                flp_cc_d:           dictionary, correlation after alignment for aligned and reversed version of each slice
                flp_imgs:           list, aligned images.
    '''

    # Downscale images ----------------------------------------------------------------------------------------------------------------------
    flp_imgs_orig  = np.array([downscale_local_mean(i,(flip_dwn_factor,flip_dwn_factor)) for i in stck_imgs])     
    # After running a brief optimization for time and convergence of cc settled with a downsampling of 10, for which 100 iterations is enough.
    warp_mode = cv2.MOTION_EUCLIDEAN # Select transformation mode
    number_of_iterations = 100;# Specify the number of iterations
    termination_eps = 1e-10;# Specify the threshold of the increment in the correlation coefficient between two iterations
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)# Define termination criteria

    # Inititalize list for propperly oriented images and flip tracking
    flp_imgs = [flp_imgs_orig[0,...]]
    flp_trk = [1]
    flp_cc_d = {'matched':[],'reversed':[]}

    for flp_img_test in flp_imgs_orig[1:,...]:

        # Natural comparison ----------------------------------------------------
        warp_matrix = np.eye(2, 3, dtype=np.float32) 
        try:
            (cc, warp_matrix) = cv2.findTransformECC (flp_imgs[-1],flp_img_test,warp_matrix, warp_mode, criteria,None,5)
            natural_cc = cc
        except cv2.error:
            natural_cc = 0

        # Reversed comparison ----------------------------------------------------
        warp_matrix = np.eye(2, 3, dtype=np.float32) 
        try:
            (cc, warp_matrix) = cv2.findTransformECC (flp_imgs[-1],cv2.flip(flp_img_test,flipCode=0),warp_matrix, warp_mode, criteria,None,5)
            reversed_cc = cc
        except cv2.error:
            reversed_cc = 0

        if natural_cc > reversed_cc:
            flp_imgs.append(flp_img_test) 
            flp_trk.append(1)
            flp_cc_d['matched'].append(natural_cc)
            flp_cc_d['reversed'].append(reversed_cc)
        else:       
            flp_imgs.append(cv2.flip(flp_img_test,flipCode=0))
            flp_trk.append(-1)
            flp_cc_d['matched'].append(reversed_cc)
            flp_cc_d['reversed'].append(natural_cc)

    return flp_trk,flp_cc_d,flp_imgs

def correct_flip(flp_trk,stck_imgs,map_imgs):

    '''
    Correct flip to get all images propperly oriented
    INPUTs:     flp_trk:        list, result of flipping assesment relative to first slice. 1 means aligned and -1 reversed. 
                stck_imgs:      array, zoomed-in images containing channel with ubiquitous labeling of region of interest. 
                                Images must be sorted from anterior to posterior location. [n_images,y,x]
                map_imgs:       array, zoomed-out images sorted from anterior to posterior location [n_images,y,x]
    OUTPUTs:    stck_imgs_flp:  array, oriented zoomed-in images
                map_imgs_flp:   array, oriented zoomed-out images
    '''

    
    stck_imgs_flp = []
    map_imgs_flp = []
    for img_idx in range(len(flp_trk)):
        if flp_trk[img_idx] == 1:
            # Save unflipped image ----------------------------
            stck_imgs_flp.append(stck_imgs[img_idx,...].copy())
            map_imgs_flp.append(map_imgs[img_idx,...].copy())
        elif flp_trk[img_idx] == -1:
            # Flip images and save ----------------------------
            stck_imgs_flp.append(cv2.flip(stck_imgs[img_idx,...].copy(),flipCode=0))
            map_imgs_flp.append(cv2.flip(map_imgs[img_idx,...].copy(),flipCode=0))
    
    stck_imgs_flp = np.array(stck_imgs_flp)
    map_imgs_flp = np.array(map_imgs_flp) 
    
    return stck_imgs_flp,map_imgs_flp

def get_masked_map(map_imgs,map_msks_crds,view_imgs=True):
    '''
    Applies masks and returns zoomed-out images 
    INPUTs: map_imgs:               array, oriented zoomed-out images                
            map_msks_crds:          dict, mapping slices to list of polygon coordinates defining mask
            view_imgs (optional):   bool, whether or not to display output in napari viewer instance

    OUTPUTs:msk_map_imgs:           array, input array after application of cleaning masks 
            msk_map_imgs_eq:        array, input array after application of cleaning masks and histogram equalization  
            n_vwr_0:                napari viewer instance, display input array with masks overlaid
    '''
    # Create images with masks applied ------------------------------------------------------------------------------------
    msk_map_imgs = []
    for s_slice in range(map_imgs.shape[0]):        
        polygon_bnds = np.array(map_msks_crds['slice_'+str(s_slice)])
        polygon_crds = np.array(polygon(polygon_bnds[:,0],polygon_bnds[:,1],map_imgs.shape[1:]))
        template_msk = np.full(map_imgs.shape[1:],fill_value=np.min(map_imgs[s_slice,...]))
        template_msk[polygon_crds[0,:],polygon_crds[1,:]] = map_imgs[s_slice,polygon_crds[0,:],polygon_crds[1,:]]
        msk_map_imgs.append(template_msk)
    msk_map_imgs = np.array(msk_map_imgs)
    # Apply CLAHE
    msk_map_imgs_eq = np.array([exposure.equalize_adapthist(i) for i in msk_map_imgs])
    # --------------------------------------------------------------------------------------------------------------------
    if view_imgs:
        # Visualize masks overlaid on original images  -----------------------------------------------------------------------------
        n_vwr_0 = napari.Viewer() 
        n_vwr_0.add_image(map_imgs,name='map images flipped',colormap='gray_r')
        for k in map_msks_crds.keys():
            recons_msk_crds = np.array(map_msks_crds[k])
            slice_idx = float(k.split('_')[1])
            slice_idx_array = np.expand_dims(np.ones(len(map_msks_crds[k]))*slice_idx,1)
            recons_msk = np.concatenate((slice_idx_array,recons_msk_crds),axis=1)
            n_vwr_0.add_shapes(name=k,data=[recons_msk],shape_type='polygon', face_color='orange',opacity=0.25)
        n_vwr_0.dims.current_step = (0,0,0) # Position slider on first slice  
    else:
        n_vwr_0 = None     

    return msk_map_imgs,msk_map_imgs_eq,n_vwr_0

def register_map_imgs(msk_map_imgs,reg_dwn_factor = 5):
    '''
    Perform sequential registration of slices using an Euclidian model.
    Transformation is performed in downscaled images for increased performance and later applied to full resolution inputs.
    INPUTs:     msk_map_imgs:   array, input array for sequential slice registration
                reg_dwn_factor: int, factor by which to downsample images before registration
    OUTPUTs:    reg_map:        array, input array after sequential registration of images
                reg_map_eq:     array, input array after sequential registration of images and histogram equalization
                warp_mat_l:     list, warp matrices used for image registration. Number of elements is one less than number of iamges given that first one remains fixed. 
    '''
    # Do serial image registration ----------------------------------------------------------------------------------------
    reg_map = [msk_map_imgs[0,...].copy()]
    warp_mat_l = []
    for map_reg in msk_map_imgs[1:,...]:
        print(f'Working on image {len(reg_map)}')
        map_ref = reg_map[-1]
        # Downscale images ------------------------------------------------------
        map_ref_dwn = downscale_local_mean(map_ref,(reg_dwn_factor,reg_dwn_factor))
        map_reg_dwn = downscale_local_mean(map_reg,(reg_dwn_factor,reg_dwn_factor))
        # Adaptive histogram equalization --------------------------------------------
        map_ref_dwn_eq = exposure.equalize_adapthist(map_ref_dwn)
        map_reg_dwn_eq = exposure.equalize_adapthist(map_reg_dwn)
        
        warp_mode = cv2.MOTION_EUCLIDEAN #Select transformation mode
        number_of_iterations = 1000;# Specify the number of iterations.
        termination_eps = 1e-10;# Specify the threshold of the increment in the correlation coefficient between two iterations
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)# Define termination criteria
        warp_matrix = np.eye(2, 3, dtype=np.float32) 
        (cc, warp_matrix) = cv2.findTransformECC (map_ref_dwn_eq,map_reg_dwn_eq,warp_matrix, warp_mode, criteria,None,5)
        # Upscale the warp_matrix ----------------------------------------------- 
        warp_matrix_up = warp_matrix.copy()
        warp_matrix_up[:,2] = warp_matrix_up[:,2]*reg_dwn_factor
        warp_mat_l.append(warp_matrix_up)
        # Register original image -----------------------------------------------
        registered_map = cv2.warpAffine(map_reg, warp_matrix_up, map_reg.shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        reg_map.append(registered_map)

    reg_map = np.array(reg_map)
    # Create adaptive histogram equalized version of registered stacks --------
    reg_map_eq = np.array([exposure.equalize_adapthist(i) for i in reg_map])
    
    return reg_map,reg_map_eq,warp_mat_l

def get_map_proj(map_imgs,map_info,map_ids,map_ant_to_post_ids,track_bounds_d,stg_to_um,down_sample=1):

    '''
    Assembly of projection displaying zoomed-out low-resultion images in stage as they were acquired
    INPUTs:     map_imgs : array, zoomed-out images sorted from anterior to posterior location [n_images,y,x]
                map_info : dict, image file specifications for zoomed-out images
                map_ids :  list, indexes of acquired zoomed-out images 
                map_ant_to_post_ids : list, manually defined indexes of zoomed-out images from anterior to posterior location
                track_bounds_d : dict, boundaries in microns defined by images taken within full microscope stage
                stg_to_um : float, size of stage units in microns 
                down_sample (optional) : int, factor by which to downsample zoomed-out image (defualt = 1) 
    OUTPUTs:    map_proj_img : array, image displaying assembly of zoomed-out images in stage    
                mapProj_pxl_size: float, pixel size in microns
    '''

    
    mapProj_pxl_size = map_info['pxl_phys_dims']*down_sample
    # Obtain utilized stage extent in pixel units --------------------------------------------------------------------------------------
    x_pxl_ext = int(np.ceil(track_bounds_d['width']/mapProj_pxl_size))
    v_pxl_ext = int(np.ceil(track_bounds_d['height']/mapProj_pxl_size))
    
    # Initialize array for projection ---------------------------------------------------------------------------------------------------
    map_proj_img = np.zeros([v_pxl_ext,x_pxl_ext])

    try:
        # Check if all slides are accounted for when specifying manual order -----------------------------------------------------------
        assert np.array_equal(np.sort(map_ant_to_post_ids),np.array(map_ids)), 'WARNING: Not all slides are accounted for' 

        # Assemble projection image -----------------------------------------------------------------------------------------------------
        for s_map,s_map_img_float in zip(reversed(map_ant_to_post_ids),np.flip(map_imgs,axis=0)):
        # Reversed order is to ensure overlapping between zoomed-out images not happenning in region of interest    
            # Downscale image  ------------------------------------------------------------------------
            s_map_img = downscale_local_mean(s_map_img_float,(down_sample,down_sample))
            # Calculate image offset in pixels -------------------------------------------------------
            s_map_v0 = (map_info['groups'][s_map]['coordinates']['y']*stg_to_um-track_bounds_d['min_v'])/mapProj_pxl_size
            s_map_x0 = (map_info['groups'][s_map]['coordinates']['x']*stg_to_um-track_bounds_d['min_x'])/mapProj_pxl_size
            s_map_crds = rectangle((s_map_v0,s_map_x0),extent=s_map_img.shape)
            map_proj_img[np.ravel(s_map_crds[0],order='F'),np.ravel(s_map_crds[1],order='F')] = np.ravel(s_map_img,order='C')

    except AssertionError as mssg:
        print(mssg)

    return map_proj_img,mapProj_pxl_size

def zoomed_apply_transform(map_ant_to_post_ids,ant_to_post_ids,map_info,stack_info,\
                           warp_mat_l,stg_to_um,flp_trk,msk_map_imgs):

    '''
    Apply transformations to layout area of zoomed-in high resolution images
    INPUTs: map_ant_to_post_ids:list, manually defined indexes of zoomed-out images from anterior to posterior location
            ant_to_post_ids:    list, manually defined indexes of zoomed-in images from anterior to posterior location    
            map_info:           dict, image file specifications for zoomed-out images           
            stack_info:         dict, image file specifications for zoomed-in images         
            warp_mat_l:         list, warp matrices used for image registration         
            stg_to_um:          float, size of stage units in microns 
            flp_trk:            list, result of flipping assesment relative to first slice. 1 means aligned and -1 reversed
            msk_map_imgs:       array, masked zoomed-out images input 
    OUTPUTs:zoomed_in_warp_d:   dict, output of applying registration to zoomed-in images
    '''

    # Initialize output dictionary ------------------------------------------------------------
    zoomed_in_warp_d = {
        'original':{'img':[],'edgeCrds':[],'fullCrds':[]},
        'transformed':{'img':[],'edgeCrds':[],'fullCrds':[]},
    }

    for order_id in range(len(map_ant_to_post_ids)):

        mp_id = map_ant_to_post_ids[order_id]
        stk_id = ant_to_post_ids[order_id]
        if order_id == 0:
            s_warp_m = np.eye(2, 3, dtype=np.float32)         
        else:
            s_warp_m = warp_mat_l[order_id-1].copy()
        # Retrieve coordinates of zoomed-in image within map image, with units of zoomed-out image pixel size--------------------
        # Units in microns with reference to full stage ------------------------------------
        mp_x0 = map_info['groups'][mp_id]['coordinates']['x'] * stg_to_um
        mp_v0 = map_info['groups'][mp_id]['coordinates']['y'] * stg_to_um
        stk_x0 = stack_info['groups'][stk_id]['coordinates']['x']* stg_to_um
        stk_v0 = stack_info['groups'][stk_id]['coordinates']['y']* stg_to_um
        stk_v_ext = stack_info['groups'][stk_id]['coordinates']['height']* stg_to_um
        stk_x_ext = stack_info['groups'][stk_id]['coordinates']['width']* stg_to_um
        # Zoomed in coordinates with reference to associated zoomed out image ------------------
        # Max extent in pixel units 
        map_ext_x = map_info['groups'][mp_id]['imgSpecs']['pxls_dims']['x']
        map_ext_y = map_info['groups'][mp_id]['imgSpecs']['pxls_dims']['y']
        # Flipping invariant values (flip happens along vertical axes) --------------------------------------------------------
        stk_pxl_x_ext =  stk_x_ext/map_info['pxl_phys_dims']
        stk_pxl_v_ext = stk_v_ext/map_info['pxl_phys_dims']
        stk_pxl_x0 = (stk_x0 - mp_x0)/map_info['pxl_phys_dims']
        # Original values that can change if image is flipped ------------------------------------------------------------------
        stk_pxl_v_orig = (stk_v0-mp_v0)/map_info['pxl_phys_dims']
        stk_pxl_v0 = stk_pxl_v_orig if flp_trk[order_id] == 1 else map_ext_y - (stk_pxl_v_orig+stk_pxl_v_ext)
        stk_pxl_x1 = stk_pxl_x0 + stk_pxl_x_ext
        stk_pxl_v1 = stk_pxl_v0 + stk_pxl_v_ext

        # Coordinates defined in circular order for propper displaying as polygon ---------------------------------------------
        stck_pxl_crds = np.array([[stk_pxl_v0,stk_pxl_v0,stk_pxl_v1,stk_pxl_v1],[stk_pxl_x0,stk_pxl_x1,stk_pxl_x1,stk_pxl_x0]])
        # Imprint original zoomed-in layout on zoomed-out image ---------------------------------------------------------------
        orig_ptch_crds = rectangle_perimeter((stk_pxl_v0,stk_pxl_x0),(stk_pxl_v1,stk_pxl_x1))
        orig_map_img = msk_map_imgs[order_id,...].copy()
        orig_map_img[np.ravel(orig_ptch_crds[0],order='F'),np.ravel(orig_ptch_crds[1],order='F')] = np.max(orig_map_img)
        # Transform original image with imprinted pattern ---------------------------------------------------------------------
        t_map_img = cv2.warpAffine(orig_map_img, s_warp_m, orig_map_img.shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        # Transform coordinate set defining zoomed-in layout ------------------------------------------------------------------
        stck_pxl_crds_inp = np.append(np.flip(stck_pxl_crds,axis=0),np.ones((1,4)),axis=0)
        t_stck_pxl_crds_0 = np.dot(cv2.invertAffineTransform(s_warp_m),stck_pxl_crds_inp)
        t_stck_pxl_crds = np.flip(t_stck_pxl_crds_0,axis=0)
        # Polygon coordinates - function return it capped for negative values
        orig_msk_crds = np.array(polygon(stck_pxl_crds[0,:],stck_pxl_crds[1,:],shape = (map_ext_y,map_ext_x)))
        t_msk_crds = np.array(polygon(t_stck_pxl_crds[0,:],t_stck_pxl_crds[1,:],shape = (map_ext_y,map_ext_x)))
        # Populate output dict --------------------------------------------------------------------------------
        zoomed_in_warp_d['original']['img'].append(orig_map_img)
        zoomed_in_warp_d['transformed']['img'].append(t_map_img)
        zoomed_in_warp_d['original']['edgeCrds'].append(stck_pxl_crds)
        zoomed_in_warp_d['transformed']['edgeCrds'].append(t_stck_pxl_crds)
        zoomed_in_warp_d['original']['fullCrds'].append(orig_msk_crds)
        zoomed_in_warp_d['transformed']['fullCrds'].append(t_msk_crds)

    
    return zoomed_in_warp_d

def get_zoomed_intersect(zoomed_in_warp_d,subselect_idx = None):
    '''
    Find intersection of transformed zoomed-in masks  
    INPUTs: zoomed_in_warp_d:           dict, output of applying registration to zoomed-in images
            subselect_idx (optional):   array, indexes of subset of slices to consider or None if all slices are used     
    OUTPUTs:t_int_bbox:                 array, coordinates of rectangle fully contained in intersection of transformed masks                      
            t_msks_int_cntr:            array, contour of intersection of transformed masks
    '''

    # Obtain extent from original images --------------------------------------------------------
    img_ext = np.array([[i.shape[0],i.shape[1]] for i in zoomed_in_warp_d['original']['img']])
    img_ext_y_uniq = np.unique(img_ext[:,0])
    img_ext_x_uniq = np.unique(img_ext[:,1])
    
    try:        
        assert img_ext_y_uniq.size==1 and img_ext_x_uniq.size==1, 'Dimensions are not consistent across input images '
        map_ext_y = img_ext_y_uniq[0]
        map_ext_x = img_ext_x_uniq[0]
        # Get masks of transformed items and find intersection -------------------------
        t_msks_l = []
        for e_s_crd in zoomed_in_warp_d['transformed']['fullCrds']: 
            s_tmpl_img = np.full(shape=(map_ext_y,map_ext_x),fill_value=False)
            s_tmpl_img[e_s_crd[0,:],e_s_crd[1,:]] = True
            t_msks_l.append(s_tmpl_img)
        t_msks = np.array(t_msks_l)
        # Deal with subselecting images for performing intersectional selection-------- 
        if subselect_idx is None:
            t_msks_int = np.all(t_msks,axis=0)
        else:
            t_msks_int = np.all(t_msks[subselect_idx,...],axis=0)
        # Retrieve contour of intersectional mask -------------------------------------
        t_msks_int_cntr = np.array(measure.find_contours(t_msks_int))[0,...].T


        # Iteratively find first box fully contained within intersectional mask -----------------------------
        # Start with bounding box ---------------------------------------------------------------------------
        t_int_props = measure.regionprops(t_msks_int.astype('int'))
        t_int_bbox = np.reshape(t_int_props[0].bbox,[2,2],order='C')
        inters_n_el = np.sum(t_msks_int)
        elements_out = True
        # Decrease bounding box dimensions until fully contained 
        while elements_out:
            t_int_bbox = t_int_bbox + np.array([[1],[-1]])
            temp_rect_box = np.full(shape=t_msks_int.shape,fill_value=False)
            temp_rect_crds = rectangle(start=t_int_bbox[0,:],end=t_int_bbox[1,:],shape=t_msks_int.shape)
            temp_rect_box[temp_rect_crds[0],temp_rect_crds[1]] = True
            isol_n_el = np.sum(np.logical_xor(temp_rect_box,t_msks_int))
            shared_n_el = np.sum(np.logical_and(temp_rect_box,t_msks_int))
            if isol_n_el + shared_n_el == inters_n_el: elements_out = False
    except AssertionError as mssg:
        t_int_bbox = None
        t_msks_int_cntr = None
        print(mssg)
    
    return t_int_bbox,t_msks_int_cntr

def align_zoomed_in(t_int_bbox,map_info,stack_info,map_ant_to_post_ids,ant_to_post_ids,zoomed_img_mmap,zoomed_in_warp_d,save_p,flp_trk):
    '''
    Extracts intersection area from each image and re-orient them. 
    Result for each slice is saved as separate file within 'zoomed_in_align_l' folder 
    INPUTs: t_int_bbox:             array, coordinates of rectangle fully contained in intersection of transformed masks 
            map_info:               dict, image file specifications for zoomed-out images
            stack_info:             dict, image file specifications for zoomed-in images
            map_ant_to_post_ids:    list, manually defined indexes of zoomed-out images from anterior to posterior location             
            ant_to_post_ids:        list, manually defined indexes of zoomed-in images from anterior to posterior location    
            zoomed_img_mmap:        array, zoomed-in high-resolution images [n_images,n_channels,n_planes,y,x]. Images must be sorted from anterior to posterior location.       
            zoomed_in_warp_d:       dict, output of applying registration to zoomed-in images        
            save_p:                 PosixPath, saving path             
            flp_trk:                list, result of flipping assesment relative to first slice. 1 means aligned and -1 reversed. 
    '''
    
    # Define path for saving ------------------------------------------------------------------------------------------------------------
    zoomed_in_al_p = os.path.join(save_p,'zoomed_in_align_l')
    if not(os.path.isdir(zoomed_in_al_p)): os.mkdir(zoomed_in_al_p)
    # Extract zoomed in section and warp -------------------------------------------------------------------------------------------------    
    # Obtain corners of intersectional box - Get it manually to ensure circular order 
    int_box_x = [t_int_bbox [0,1], t_int_bbox[1,1],t_int_bbox[1,1],t_int_bbox [0,1]] 
    int_box_v = [t_int_bbox[0,0],t_int_bbox[0,0],t_int_bbox[1,0],t_int_bbox[1,0]]
    int_box_crds = np.array([int_box_v,int_box_x]).T
    map_to_stck = map_info['pxl_phys_dims'] / stack_info['pxl_phys_dims']
    
    for e_s_stk_id in range(len(map_ant_to_post_ids)):
        # Calculate position in new coordinate system by finding distance of each point with bounding lines
        all_edge_crds = zoomed_in_warp_d['transformed']['edgeCrds'][e_s_stk_id]
        # Reference to lower left coordinate - it is going to aid later indexing into zoomed in image
        orig_pnt = all_edge_crds[:,0]
        v_extrm_pnt = all_edge_crds[:,3] 
        h_extrm_pnt = all_edge_crds[:,1] 
        # Check with left out coordinate for reassurance ---------------------------------------------------------- 
        test_pnt = all_edge_crds[:,2]

        h_crd_test =  np.abs((v_extrm_pnt[1]-orig_pnt[1])*test_pnt[0]-(v_extrm_pnt[0]-orig_pnt[0])*test_pnt[1] +\
                            v_extrm_pnt[0]*orig_pnt[1] - v_extrm_pnt[1]*orig_pnt[0])/ \
                            np.linalg.norm(v_extrm_pnt-orig_pnt) 

        v_crd_test =  np.abs((h_extrm_pnt[1]-orig_pnt[1])*test_pnt[0]-(h_extrm_pnt[0]-orig_pnt[0])*test_pnt[1] +\
                            h_extrm_pnt[0]*orig_pnt[1] - h_extrm_pnt[1]*orig_pnt[0])/ \
                            np.linalg.norm(h_extrm_pnt-orig_pnt) 
        # Rescale coordinates to pixel units in zoomed in image ------------------
        h_crd_test_rscl = h_crd_test * map_to_stck
        v_crd_test_rscl = v_crd_test * map_to_stck
        # Retrieve expected coordinates from image size ---------------------------
        stck_x_pxls = stack_info['groups'][ant_to_post_ids[e_s_stk_id]]['imgSpecs']['pxls_dims']['x']
        stck_y_pxls = stack_info['groups'][ant_to_post_ids[e_s_stk_id]]['imgSpecs']['pxls_dims']['y']
        # Stringest test to move on, single pixel tolerace -----------------------------------------
        try:
            assert np.abs(h_crd_test_rscl-stck_x_pxls)<1 and  np.abs(v_crd_test_rscl-stck_y_pxls)<1, 'Coordinate system transformation is giving an error'  
            # Get transformed extraction coordinates 
            t_ext_crds = []
            for s_int_box  in int_box_crds:
                
                h_crd_loc =  np.abs((v_extrm_pnt[1]-orig_pnt[1])*s_int_box[0]-(v_extrm_pnt[0]-orig_pnt[0])*s_int_box[1] +\
                            v_extrm_pnt[0]*orig_pnt[1] - v_extrm_pnt[1]*orig_pnt[0])/ \
                            np.linalg.norm(v_extrm_pnt-orig_pnt) 

                v_crd_loc =  np.abs((h_extrm_pnt[1]-orig_pnt[1])*s_int_box[0]-(h_extrm_pnt[0]-orig_pnt[0])*s_int_box[1] +\
                            h_extrm_pnt[0]*orig_pnt[1] - h_extrm_pnt[1]*orig_pnt[0])/ \
                            np.linalg.norm(h_extrm_pnt-orig_pnt) 

                t_ext_crds.append([v_crd_loc* map_to_stck,h_crd_loc* map_to_stck])

            t_ext_crds = np.array(t_ext_crds)

        except AssertionError as mssg:
            print(mssg)

        # Strategy extract image and rotate until to make it vertical, then remove the padding edges
        #Fine tuning of pixel dimensions matching 
        ext_msk_0 = polygon2mask((stck_y_pxls,stck_x_pxls),t_ext_crds).astype('int')
        # Obtain angle required to rotate image 
        # Obtain angle with vertical axes keeping coordinate system of image
        dir_vect =  t_ext_crds[3,:] - t_ext_crds[0,:] 
        dir_angle_rad = -np.arctan(dir_vect[1]/dir_vect[0])
        dir_angle_deg = np.degrees(dir_angle_rad)        
        # Get slice from memmap file
        zoomed_img = zoomed_img_mmap[e_s_stk_id,...].copy()    
        # Dealing with flipping 
        if flp_trk[e_s_stk_id] == 1:
            orig_stck_img = zoomed_img
        elif flp_trk[e_s_stk_id] == -1:     
            orig_stck_img = np.flip(zoomed_img,axis=2)
        
        isol_stck_img = orig_stck_img*ext_msk_0 # Casting for channels and z_steps dimensions
        # Use max projection of first channel for getting bounding box
        lbl_stck_img = measure.label(np.max(isol_stck_img[0,...],axis=0)>0)
        props_stck_img = measure.regionprops(lbl_stck_img)
        
        try:
            # Get first bounding box to perform centered rotation ----------------------------------------
            assert len(props_stck_img) == 1, 'Can not move on because more than one block was detected'
            cmp_stck_img = props_stck_img[0]
            bbox_stck_img = cmp_stck_img.bbox        
            extr_stck_img = isol_stck_img[:,:,bbox_stck_img[0]:bbox_stck_img[2],bbox_stck_img[1]:bbox_stck_img[3]]
            aligned_stck_img = ndimage.rotate(extr_stck_img,dir_angle_deg,axes=(2,3))
            # Save zoomed_in_aligned big array into hard disk -----------------------------------------------------------------------------------
            # -------------------------------------------------------------------------------------------------------------------------------------    
            # Saving as separate files for each slice given inhomogeneous shape -------------------------------------------------------------------
            np.save(os.path.join(zoomed_in_al_p,f'zoomed_in_align_slice_{e_s_stk_id}.npy'),aligned_stck_img)

        except AssertionError as mssg:
            print(mssg)

def zoomed_in_dims_match(save_p): 
    '''
    Match dimensions of zoomed-in re-oriented images. Result is saved as memory mapped .tif file 'zommed_align_matched.tif'
    INPUTs: save_p: PosixPath, saving path 
    '''
    # Load zoomed_in_aligned list --------------------------------------------------------------------------------------------------------
    zoomed_in_al_p = os.path.join(save_p,'zoomed_in_align_l')
    zoomed_in_align = []
    for e_s_slc in range(len(os.listdir(zoomed_in_al_p))):
        zoomed_in_align.append(np.load(os.path.join(zoomed_in_al_p,f'zoomed_in_align_slice_{e_s_slc}.npy')))

    # Clean up to take dimensions to minimal available ---------------------------------------------------------------------------------
    all_dims = np.array([list(i.shape[2:]) for i in zoomed_in_align])
    half_min_v,half_min_h = (np.min(all_dims,axis=0)/2).astype('int')
    
    # Instantiate memory mapped file - Dealing with huge file size
    mmap_out_shape = (len(zoomed_in_align),zoomed_in_align[0].shape[0],zoomed_in_align[0].shape[1],int(half_min_v*2),int(half_min_h*2))
    zommed_align_matched = memmap(os.path.join(save_p,'zommed_align_matched.tif'),shape=mmap_out_shape,dtype=zoomed_in_align[0].dtype)
    
    for id_algn_img, algn_img in enumerate(zoomed_in_align):
        v_center,h_center = (all_dims[id_algn_img,:]/2).astype('int')
        crop_align = algn_img[:,:,v_center-half_min_v:v_center+half_min_v,\
                            h_center-half_min_h:h_center+half_min_h].copy()
    
        zommed_align_matched[id_algn_img,...] = crop_align 
        zommed_align_matched.flush()

def visualize_stcks_vol(zommed_align_matched_mmap,stack_info,ant_to_post_ids,z_offset,display_mode,skip_idx):
    '''
    2D and 3D visualization of aligned zoomed-in high-resolution images
    INPUTs:     zommed_align_matched_mmap:  array, zoomed-in aligned images with matching dimensions [n_images,n_channels,n_planes,y,x]
                stack_info:                 dict, image file specifications for zoomed-in images
                ant_to_post_ids:            list, manually defined indexes of zoomed-in images from anterior to posterior location    
                z_offset:                   int, spacing in microns between mounted slices in the antero-posterior axis
                display_mode:               string, chose between following options
                                                    'zProj_raw': slices displayed with no separation in z-axis
                                                    'zProj_block': volumetric representation achieved by spacing between slices with zero values
                skip_idx:                   list, indexes of slices to skip when creating visualization
    OUTPUTs:    n_vwr_x:                    napari viewer instance
    '''


    # Create boolean array for slices sub-selection-----------------------------------------------------------------------------------------
    keep_bool = np.full(zommed_align_matched_mmap.shape[0],fill_value=True)
    keep_bool[skip_idx] = False

    # Obtain common specs for different kind of displays -------------------------------------------------------------------------------------------------------------------------------------    
    # Obtain z-specs from one of the images - Assumption that it is going to be the same for all 
    z_step = np.unique(np.diff(stack_info['groups'][ant_to_post_ids[0]]['imgSpecs']['acq_specs']['z_abs']))[0]
    z_pxl_size = stack_info['groups'][ant_to_post_ids[0]]['imgSpecs']['phys_dims']['z']
    n_planes = stack_info['groups'][ant_to_post_ids[0]]['imgSpecs']['pxls_dims']['z']
    lat_pxl_size = stack_info['groups'][ant_to_post_ids[0]]['imgSpecs']['phys_dims']['x']

    # DISPLAY ---------------------------------------------------------------------------------------------------------------   
    n_channels = zommed_align_matched_mmap.shape[1]
    spec_cmap = ['bop blue','Green','Red'] if n_channels == 3 else ['bop blue','Red']
    spec_names = ['blue','green','red'] if n_channels == 3 else ['blue','red']
    
    try:
        assert np.allclose(z_step,z_pxl_size,atol = 0.01), 'Warning, z_step does not match z_pxl_size, unexpected results while rendering'
        
        if display_mode == 'zProj_raw':       
            n_vwr_0 = napari.Viewer() 
            n_vwr_0.add_image(np.max(zommed_align_matched_mmap,axis=2)[keep_bool,...],channel_axis=1,colormap=spec_cmap,name=spec_names,scale=[z_pxl_size*n_planes,lat_pxl_size,lat_pxl_size]) 
            n_vwr_x = n_vwr_0   

        elif display_mode == 'zProj_block':
            # Introducing downsizing to avoid memory issues in laptop when performing napari volumetric display
            vis_dwn_fctr = 2 # Lateral downscaling factor
            n_pxls_offset = int(np.round(z_offset/(z_pxl_size*n_planes)))
            vh_dims_0 = np.array(zommed_align_matched_mmap.shape[3:])/vis_dwn_fctr
            vh_dims = np.array([int(np.ceil(vh_dims_0[0])),int(np.ceil(vh_dims_0[1]))])
            zommed_align_matched_depth = np.empty([0,n_channels,vh_dims[0],vh_dims[1]])
            zommed_align_nan_fill = np.zeros((n_pxls_offset,n_channels,vh_dims[0],vh_dims[1]))

            for e_zoomed_align_idx,e_zoomed_align in enumerate(zommed_align_matched_mmap):        
                if keep_bool[e_zoomed_align_idx]:
                    e_zoomed_align_proj = np.max(e_zoomed_align,axis=1)
                else:
                    e_zoomed_align_proj = np.zeros_like(np.max(e_zoomed_align,axis=1)) 

                e_zoomed_align_proj_dwn = downscale_local_mean(e_zoomed_align_proj,(1,vis_dwn_fctr,vis_dwn_fctr))
                try:
                    assert np.array_equal((e_zoomed_align_proj_dwn.shape[1:]),vh_dims), 'Dimensions are not consistent'
                    zommed_align_matched_depth = np.append(zommed_align_matched_depth,[e_zoomed_align_proj_dwn],axis=0)
                    zommed_align_matched_depth = np.append(zommed_align_matched_depth,zommed_align_nan_fill,axis=0)
                except AssertionError as mssg:
                    print(mssg)

            n_vwr_1 = napari.Viewer() 
            n_vwr_1.add_image(zommed_align_matched_depth,channel_axis=1,colormap=spec_cmap,name=spec_names,scale=[z_pxl_size*n_planes,lat_pxl_size*vis_dwn_fctr,lat_pxl_size*vis_dwn_fctr]) 
            n_vwr_1.dims.ndisplay = 3  
            n_vwr_x = n_vwr_1  
            
        # Add scale bar ---------------------
        n_vwr_x.scale_bar.visible = True
        n_vwr_x.scale_bar.unit = "um"
        n_vwr_x.scale_bar.font_size = 18
        n_vwr_x.scale_bar.ticks = False
        n_vwr_x.scale_bar.colored = True
        n_vwr_x.scale_bar.color = 'darkorange'

    except AssertionError as mssg:
        n_vwr_x = None
        print(mssg)


    return n_vwr_x





