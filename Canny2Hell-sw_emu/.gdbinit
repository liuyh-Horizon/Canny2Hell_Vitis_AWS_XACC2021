set substitute-path '/home/centos/hls_project/Canny2Hell/Emulation-SW/krnl_canny.build/link/int/xo/canny_accel/canny_accel/cpu_sources/' '/home/centos/hls_project/Canny2Hell/src/'
set substitute-path '/home/centos/hls_project/Canny2Hell/Emulation-SW/krnl_canny.build/link/int/xo/edgetracing_accel/edgetracing_accel/cpu_sources/' '/home/centos/hls_project/Canny2Hell/src/'
source /opt/xilinx/xrt/share/appdebug/appdebug.py
handle SIGUSR1 nostop pass
handle SIGTERM nostop pass
