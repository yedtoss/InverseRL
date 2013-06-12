TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH +=/usr/local/include/eigen3 /usr/local/include/eigen3/unsupported
INCLUDEPATH +=/ari1/Test2/trunk/src/core /ari1/Test2/trunk/src/export/inc /ari1/Test2/trunk/src/core /ari1/Test2/trunk/src/algorithms /ari1/Test2/trunk/src/geometry /ari1/Test2/trunk/src/models /ari1/Test2/trunk/src/statistics /ari1/Test2/trunk/src/environments /home/yedtoss/Documents/article/cpp/src /home/yedtoss/Documents/article/cpp/src /usr/local/include

INCLUDEPATH +=/home/yedtoss/Documents/Ipopt-3.10.2/build/include/coin/ThirdParty /home/yedtoss/Documents/Ipopt-3.10.2/build/include/coin

QMAKE_CXXFLAGS += -DMAKE_MAIN -DUSE_DOUBLE -O3

LIBS += -L/home/yedtoss/Documents/article/cpp/src -L/ari1/Test2/trunk/src/lib -L/ari1/Test2/trunk/src/export/lib -L/usr/local/lib -lmpfr -lgmp -llbfgs -lsmpl -lalglib -lboost_random

LIBS += `PKG_CONFIG_PATH=/home/yedtoss/Documents/Ipopt-3.10.2/build/lib/pkgconfig:/home/yedtoss/Documents/Ipopt-3.10.2/build/share/pkgconfig: /usr/bin/pkg-config --libs ipopt` -Wl,--rpath -Wl,/home/yedtoss/Documents/Ipopt-3.10.2/build/lib

SOURCES += \
    RL.cpp

HEADERS += \
    STEnvironment.h \
    TDLearning.h \
    VIAMG.h \
    AppBlackjack.h \
    AppEnvironment.h \
    ApproximatePolicyIteration.h \
    Approximator.h \
    AppTTT.h \
    FittedQEval.h \
    FittedQIteration.h \
    FNNApproximator.h \
    GreedyGQ.h \
    IRL.h \
    LinearApproximator.h \
    LRP.h \
    LSPI.h \
    LSTDQ.h \
    MatlabWrapper.h \
    MG.h \
    mpreal2.h \
    MPRealSupport \
    MyTypes.h \
    NNApproximator.h \
    OffPac.h \
    Optimize.h \
    Random2.h \
    Utils.h

OTHER_FILES += \
    ipopt.opt \
    opt

