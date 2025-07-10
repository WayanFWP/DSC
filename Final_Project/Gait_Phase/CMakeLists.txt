cmake_minimum_required(VERSION 3.16)
project(Gait_Phase VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Charts PrintSupport)

set(SOURCES
    ./main.cpp
    src/mainwindow.cpp
    src/model.cpp
    src/qcustomplot.cpp
)
set(HEADERS
    include/mainwindow.h
    include/dataset.hpp
    include/model.hpp
    include/qcustomplot.h
    src/mainwindow.ui
)


set(PROJECT_SOURCES ${SOURCES} ${HEADERS})

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(Gait_Phase
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}

    )
else()
    if(ANDROID)
        add_library(Gait_Phase SHARED ${PROJECT_SOURCES})
    else()
        add_executable(Gait_Phase ${PROJECT_SOURCES})
    endif()
endif()

target_include_directories(Gait_Phase PRIVATE include)
target_link_libraries(Gait_Phase PRIVATE Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::Charts Qt${QT_VERSION_MAJOR}::PrintSupport)

if(${QT_VERSION} VERSION_LESS 6.1.0)
    set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.Gait_Phase)
endif()

set_target_properties(Gait_Phase PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS Gait_Phase
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(Gait_Phase)
endif()
