FROM ubuntu:22.04
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get -y install --no-install-recommends g++ clang python3 cmake libboost-dev libboost-context-dev doxygen gfortran make perl python3-pip mpich libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0 libfontconfig1 libgl1-mesa-glx libdbus-1-dev tar gzip unzip
RUN apt-get update && apt-get install -y vim

RUN mkdir /root/simgrid_inst
RUN mkdir /root/simgrid_inst/simgrid-v3.35
WORKDIR /root/simgrid_inst/simgrid-v3.35
COPY ./simgrid-v3.35/ /root/simgrid_inst/simgrid-v3.35/

RUN cmake -DCMAKE_INSTALL_PREFIX=/opt/simgrid .
RUN make -j8
RUN make install

RUN pip3 install --upgrade pip setuptools

RUN pip3 install networkx numpy pyyaml torch torchvision matplotlib pandas openpyxl line_profiler PyQt5

ENV PATH /opt/simgrid/bin/:$PATH
WORKDIR /root/workspace

COPY gurobi12.0.3_linux64.tar.gz /tmp/

# Gurobiをインストール
RUN cd /tmp \
    && tar xvfz gurobi12.0.3_linux64.tar.gz \
    && rm gurobi12.0.3_linux64.tar.gz \
    && /tmp/gurobi1203/linux64/install.sh -f /opt/gurobi1203

# Gurobiの環境変数を設定
ENV GUROBI_HOME="/opt/gurobi1203/linux64"
ENV PATH="${GUROBI_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"

# PythonのGurobiライブラリをインストール
RUN pip3 install "${GUROBI_HOME}/python/gurobipy"

# Gurobiのライセンスファイルをコンテナ内にコピー
# このライセンスファイルは、ローカルPCの適切な場所に配置しておく必要があります
COPY gurobi.lic /opt/gurobi1203/linux64/
