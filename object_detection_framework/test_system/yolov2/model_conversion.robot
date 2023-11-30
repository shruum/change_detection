*** Settings ***
Documentation    Facility to test model conversion and inference of YoloV2.

Suite Setup    Prepare environment    ${TEST_DATA}
Library    OperatingSystem
Library    Process
Library    String
Resource    ${KW_DIR}/KW_json.robot
Resource    ${KW_DIR}/KW_onnx.robot
Resource    ${KW_DIR}/KW_setup.robot

*** Variables ***
${TEST_DATA}    /data/nie/teams/arl/system_tests_data/object_detection_framework/yolov2
${BIN_DIR}    ${HOME_DIR}/build/bin
${OUT_DIR}    ${HOME_DIR}
${MODEL}    yolov2
${CONFIG_FILE}    configs/yolo/darknet19_yolov2_voc0712.yaml
${TEST_OUT}    ${CURDIR}/tdo
${TEST_IN}    ${TEST_DATA}/VOC2007/JPEGImages
${HOME_DIR}    ${CURDIR}/../..
${KW_DIR}    ${CURDIR}/../keywords

*** Test Cases ***

Convert PyTorch model into ONNX using modeltoonnx to yolov2
    [Tags]    onnx_converter
    ${cmd}=    catenate    SEPARATOR=
    ...    ${HOME_DIR}/modeltoonnx.py
    ...    \ --config_file ${HOME_DIR}/${CONFIG_FILE}
    ...    \ --export_path=${TEST_OUT}/${MODEL}.onnx
    ...    \ --pretrained=True
    ...    \ --pretrained_path=${MODEL_FILE}
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}    values=False
    Generated onnx model should be valid    ${MODEL}.onnx    backbone_nr=108    header_line_nr=26    nms_nr=1

Engine generation
    [Tags]    engine
    ${cmd}=    Set Variable    LD_LIBRARY_PATH=${BIN_DIR}:$LD_LIBRARY_PATH ${BIN_DIR}/trtexec --onnx ${TEST_OUT}/${MODEL}.onnx --engine=${TEST_OUT}/${MODEL}.engine
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    ${rc}    0

Inference with onnx engine and trtexec
    [Tags]    inference    trtexec
    ${cmd}=    Set Variable
    ...    LD_LIBRARY_PATH=${BIN_DIR}:$LD_LIBRARY_PATH ${BIN_DIR}/trtexec --engine=${TEST_OUT}/${MODEL}.engine --imageDirectoryIn ${TEST_IN} --printOutput
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    ${rc}    0

Inference with onnx engine and trtexec_json
    [Tags]    inference    trtexec_json
    Create Directory    ${TEST_OUT}/detections
    ${rc}    ${log}=    Run engine with trtexec_json
    ...    executor=${BIN_DIR}/trtexec_json   model=${MODEL}    threshold=0.1    test_out=${TEST_OUT}    test_data=${TEST_IN}
    Log    ${log}
    Should Be Equal As Integers    ${rc}    0
    ${expected_list}=    Create List
    Run Keyword And Continue On Failure    Check detections in json
    ...    id=000001    expected_categories=${expected_list}    results_location=${TEST_OUT}/detections
    ${expected_list}=    Create List    8
    Run Keyword And Continue On Failure    Check detections in json
    ...    id=000004    expected_categories=${expected_list}    results_location=${TEST_OUT}/detections
    ${expected_list}=    Create List
    Run Keyword And Continue On Failure    Check detections in json
    ...    id=000009    expected_categories=${expected_list}    results_location=${TEST_OUT}/detections

Inference with PyTorch
    [Tags]    inference    pytorch
    ${cmd}=    catenate    SEPARATOR=
    ...    ${HOME_DIR}/test.py
    ...    \ --config-file ${HOME_DIR}/${CONFIG_FILE}
    ...    \ --ckpt ${MODEL_FILE}
    ...    \ --eval_only
    ...    \ OUTPUT_DIR ../../outputs LOGGER.DEBUG_MODE True TEST.CONFIDENCE_THRESHOLD 0.01
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    ${rc}    0    msg=pytorch returned error code ${rc}    values=False
    Run Keyword And Continue On Failure    Should Match Regexp    ${log}    dog *: 1.0000
    Should Match Regexp    ${log}    person *: 1.0000
