from pathlib import Path
import os


#MODELS_DIR = Path(r"C:/Users/vadla/Retina_App/MODELS_DIR")
#REPORTS_DIR = Path(r"C:/Users/vadla/Retina_App/REPORTS_DIR")

IMAGE_SIZE = (384, 384)

CLASS_NAMES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]

CLINICAL_NOTES = {
    "AMD": "Age-related macular degeneration. Affects outer retina/RPE. Look for drusen, RPE irregularity, subretinal fluid.",
    "CNV": "Choroidal neovascularization. Abnormal vessels from choroid. Look for sub-RPE fluid, hyperreflective lesion.",
    "CSR": "Central serous retinopathy. Serous detachment at macula. Look for dome-shaped detachment, subretinal fluid.",
    "DME": "Diabetic macular edema. Fluid in macula from diabetes. Look for intraretinal cysts, increased thickness.",
    "DR": "Diabetic retinopathy. Microvascular damage. Look for microaneurysms, hemorrhages, hard exudates.",
    "DRUSEN": "Drusen deposits beneath RPE. Early AMD marker. Monitor for progression to wet AMD.",
    "MH": "Macular hole. Full-thickness foveal defect. Look for surrounding cuff of subretinal fluid.",
    "NORMAL": "Normal retinal structure. All layers intact, no fluid or lesions detected.",
}