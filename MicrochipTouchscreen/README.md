# MicrochipTouchscreen

This project is used to visualize raw touch data captured by Microchip touchscreen development kit.


## System Requirements

* windows
* cmake
* visual studio


##  Usage

* clone the repo to your local machine

* update submodules by

        git submodule update --init --recursive

* config the project with cmake gui

        open cmake gui
        set source to the REPO_ROOT
        set build binaries to REPO_ROOT/build
        click configure
        click generate

* open generated Visual Studio soluton and compile with both Debug and Release mode

* run MicrochipTouchScreen

        A window will appear showing the realtime raw heatmap


## maXTouch Studio

The official software to control the Microchip touchscreen development kit is [maXTouch Studio](https://developerhelp.microchip.com/xwiki/bin/view/applications/touch-gesture/maxtouch/touchscreen-interface-maxtouch-studio-lite/), which provide full access of all parameters of the touchscreen. You can also download the software from here: 
    
    链接: https://pan.baidu.com/s/1JQ4kNVuDMKBEewKK2E8BZQ?pwd=i6iy 提取码: i6iy 


### Visualize raw touch data

    Tools -> Graphical Debug Viewer

        Source: Debug Interface
        Mutual Cap: Delta8

    Actions -> Green Arrow


### Reset touchscreen

Device is initialized on power up. You can manually reset by

    Device -> Calibrate

