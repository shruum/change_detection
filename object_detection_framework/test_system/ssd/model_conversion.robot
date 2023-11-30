*** Settings ***
Documentation    Facility to test model conversion and inference of ssd

Suite Setup    Prepare environment
Library    Collections
Library    OperatingSystem
Library    Process
Library    String
Resource    ${KW_DIR}/KW_data_utility.robot
Resource    ${KW_DIR}/KW_inference.robot
Resource    ${KW_DIR}/KW_onnx.robot

*** Variables ***
${TEST_OUT}    ${CURDIR}/tdo
${KW_DIR}    ${CURDIR}/../keywords
@{MODEL_LIST}    mobilenet    resnet18    peleenet    darknet
@{DATA_TYPE}    voc    coco
@{TEST_CASES}

*** Test Cases ***
Test mobilenet voc
    [Tags]    mobilenet    voc
    Execute test for    mobilenet    voc

Test mobilenet coco
    [Tags]    mobilenet    coco
    Execute test for    mobilenet    coco

Test resnet18 voc
    [Tags]    resnet18    voc
    Execute test for    resnet18    voc

Test resnet18 coco
    [Tags]    resnet18    coco
    Execute test for    resnet18    coco

# disable peleenet and darknet system test
# since the new model file is not ready yet
# need to replace the model file and enable
# these two tests when the new model is ready
#Test peleenet voc
#    [Tags]    peleenet    voc
#    Execute test for    peleenet    voc
#
#Test peleenet coco
#    [Tags]    peleenet    voc
#    Execute test for    peleenet    coco
#
#Test darknet voc
#    [Tags]    darknet    voc
#    Execute test for    darknet    voc
#
#Test darknet coco
#    [Tags]    darknet    coco
#    Execute test for    darknet    coco

*** Keywords ***

Prepare environment
    :FOR    ${model}    IN    @{MODEL_LIST}
    \    Append model    ${model}
    :FOR    ${testcase}    IN    @{TEST_CASES}
    \    Create Directory    ${TEST_OUT}/${testcase}

Append model
    [Arguments]    ${model}
    :For    ${dtype}    IN    @{DATA_TYPE}
    \    Append To List    ${TEST_CASES}    ${model}_${dtype}

Execute test for
    [Arguments]    ${model}    ${dtype}
    Convert PyTorch model into ONNX ssd    ${model}    ${dtype}
    Engine generation    ${model}    ${dtype}
    Inference with onnx engine and trtexec_json    ${model}    ${dtype}
    Check result    ${model}    ${dtype}