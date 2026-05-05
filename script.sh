# TRAIN
python main.py train # SegFormer
python main.py train --model mobilenetv4 # MobileNetV4
python main.py train --model mobilenetv4 --loss focal --gamma 2.0 # MobileNetV4 with focal loss
python main.py train --loss focal --gamma 2.0 # SegFormer with focal loss

# CALIBRATE
python main.py calibrate --calibration_tecnique matrix_scaling # SegFormer default
python main.py calibrate --calibration_tecnique matrix_scaling --num_epochs 50
python main.py calibrate --calibration_tecnique matrix_scaling --num_epochs 100
python main.py calibrate --calibration_tecnique temperature_scaling
python main.py calibrate --calibration_tecnique temperature_scaling --num_epochs 30
# Focal loss
python main.py calibrate --calibration_tecnique temperature_scaling --num_epochs 30 --checkpoint weights/segformer_focal_gamma2.0.pth
python main.py calibrate --calibration_tecnique matrix_scaling --num_epochs 100 --checkpoint weights/segformer_focal_gamma2.0.pth
# MobileNetV4
python main.py calibrate --model mobilenetv4 --calibration_tecnique matrix_scaling --num_epochs 100 --checkpoint weights/mobilenetv4.pth
python main.py calibrate --model mobilenetv4 --calibration_tecnique temperature_scaling --num_epochs 30 --checkpoint weights/mobilenetv4.pth
python main.py calibrate --model mobilenetv4 --calibration_tecnique matrix_scaling --num_epochs 100 --checkpoint weights/mobilenetv4_focal_gamma2.0.pth
python main.py calibrate --model mobilenetv4 --calibration_tecnique temperature_scaling --num_epochs 30 --checkpoint weights/mobilenetv4_focal_gamma2.0.pth

# EVAL
python main.py evaluate # SegFormer

python main.py evaluate --checkpoint weights/segformer_focal_gamma2.0.pth

python main.py evaluate --calibration_tecnique matrix_scaling --calibration_params weights/segformer_calibrated_n50_matrix_scaling.pkl
python main.py evaluate --calibration_tecnique matrix_scaling --calibration_params weights/segformer_calibrated_n100_matrix_scaling.pkl
python main.py evaluate --calibration_tecnique temperature_scaling --calibration_params weights/segformer_calibrated_n15_temperature_scaling.pkl
python main.py evaluate --calibration_tecnique temperature_scaling --calibration_params weights/segformer_calibrated_n30_temperature_scaling.pkl
# Focal loss
python main.py evaluate --calibration_tecnique temperature_scaling --calibration_params weights/segformer_calibrated_n30_temperature_scaling_ckpt_segformer_focal_gamma2.pkl --checkpoint weights/segformer_focal_gamma2.0.pth
python main.py evaluate --calibration_tecnique matrix_scaling --calibration_params weights/segformer_calibrated_n100_matrix_scaling_ckpt_segformer_focal_gamma2.pkl --checkpoint weights/segformer_focal_gamma2.0.pth
### MobileNetV4
python main.py evaluate --model mobilenetv4 --checkpoint weights/mobilenetv4.pth # MobileNetV4
python main.py evaluate --model mobilenetv4 --checkpoint weights/mobilenetv4_focal_gamma2.0.pth # MobileNetV4 with focal loss
python main.py evaluate --model mobilenetv4 --calibration_tecnique matrix_scaling --calibration_params weights/mobilenetv4_calibrated_n100_matrix_scaling_ckpt_mobilenetv4.pkl --checkpoint weights/mobilenetv4.pth
python main.py evaluate --model mobilenetv4 --calibration_tecnique temperature_scaling --calibration_params weights/mobilenetv4_calibrated_n30_temperature_scaling_ckpt_mobilenetv4.pkl --checkpoint weights/mobilenetv4.pth
# Focal loss
python main.py evaluate --model mobilenetv4 --calibration_tecnique matrix_scaling --calibration_params weights/mobilenetv4_calibrated_n100_matrix_scaling_ckpt_mobilenetv4_focal_gamma2.pkl --checkpoint weights/mobilenetv4_focal_gamma2.0.pth
python main.py evaluate --model mobilenetv4 --calibration_tecnique temperature_scaling --calibration_params weights/mobilenetv4_calibrated_n30_temperature_scaling_ckpt_mobilenetv4_focal_gamma2.pkl --checkpoint weights/mobilenetv4_focal_gamma2.0.pth