import ants
from ants.internal import get_lib_fn, get_pointer_string 
from ants.core import ants_image as iio
import nibabel as nib
import os
from tempfile import mkstemp
import numpy as np


# OVERWRITE
def kelly_kapowski(s, g, w, its=45, r=0.025, m=1.5, gm_label=2, wm_label=3, **kwargs):
    """
    Compute cortical thickness using the DiReCT algorithm
    """
    if ants.is_image(s):
        s = s.clone('unsigned int')

    d = s.dimension
    outimg = g.clone() * 0.0
    kellargs = {'d': d,
                's': "[{},{},{}]".format(get_pointer_string(s),gm_label,wm_label),
                'g': g,
                'w': w,
                'c': "[{}]".format(its),
                'r': r,
                'm': m,
                'o': outimg}
    for k, v in kwargs.items():
        kellargs[k] = v
    
    print(kellargs)
    processed_kellargs = process_arguments(kellargs)
    print(processed_kellargs)
    libfn = get_lib_fn('KellyKapowski')
    libfn(processed_kellargs)

    return outimg

# OVERWRITE
def process_arguments(args):
    """
    Needs to be better validated.
    """
    p_args = []
    if isinstance(args, dict):
        for argname, argval in args.items():
            if "-MULTINAME-" in argname:
                # have this little hack because python doesnt support
                # multiple dict entries w/ the same key like R lists
                argname = argname[: argname.find("-MULTINAME-")]
            if argval is not None:
                if len(argname) > 1:
                    p_args.append("--%s" % argname)
                else:
                    p_args.append("-%s" % argname)

                if isinstance(argval, iio.ANTsImage):
                    p_args.append(_ptrstr(argval.pointer))
                elif isinstance(argval, list):
                    p = "["
                    for av in argval:
                        if isinstance(av, iio.ANTsImage):
                            av = _ptrstr(av.pointer)
                        elif str(av) == "True":
                            av = str(1)
                        elif str(av) == "False":
                            av = str(0)
                        p += av + ","
                    p += "]"

                    p_args.append(p)
                else:
                    p_args.append(str(argval))

    elif isinstance(args, list):
        for arg in args:
            if isinstance(arg, iio.ANTsImage):
                pointer_string = _ptrstr(arg.pointer)
                p_arg = pointer_string
            elif arg is None:
                pass
            elif str(arg) == "True":
                p_arg = str(1)
            elif str(arg) == "False":
                p_arg = str(0)
            else:
                p_arg = str(arg)
            p_args.append(p_arg)
    return p_args


    
# OVERWRITE 
def nifti_to_ants( nib_image ):
    """
    Converts a given Nifti image into an ANTsPy image

    Parameters
    ----------
        img: NiftiImage

    Returns
    -------
        ants_image: ANTsImage
    """
    ndim = nib_image.ndim

    if ndim < 3:
        print("Dimensionality is less than 3.")
        return None

    q_form = nib_image.get_qform()
    spacing = nib_image.header["pixdim"][1 : ndim + 1]

    origin = np.zeros((ndim))
    origin[:3] = q_form[:3, 3]

    direction = np.diag(np.ones(ndim))
    direction[:3, :3] = q_form[:3, :3] / spacing[:3]

    ants_img = ants.from_numpy(
        data = nib_image.get_fdata().astype( np.float32 ),
        origin = origin.tolist(),
        spacing = spacing.tolist(),
        direction = direction )
    
    return ants_img


def _ptrstr(pointer):
    """ get string representation of a py::capsule (aka pointer) """
    libfn = get_lib_fn("ptrstr")
    return libfn(pointer)

def save_img(img, dst, name, ref_img):
    fname = '{}/{}.nii.gz'.format(dst, name)
    niftiImg = nib.Nifti1Image(img, ref_img.affine)
    niftiImg.header['xyzt_units'] = 2  # mm
    nib.save(niftiImg, fname)
