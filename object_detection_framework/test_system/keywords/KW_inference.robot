*** Settings ***
Library    OperatingSystem
Library    String

*** Variables ***
${TEST_DATA_DIR}    /data/nie/teams/arl/system_tests_data/object_detection_framework/test_data
${HOME_DIR}    ${CURDIR}/../..
${BIN_DIR}    ${HOME_DIR}/build/bin
${TRTEXEC}    ${BIN_DIR}/trtexec
${TRTEXEC_JSON}    ${BIN_DIR}/trtexec_json
${TEST_OUT}    ${CURDIR}/tdo
&{OUTPUT_NAMES}    'mobilenet'=Output    'resnet18'=Output    'peleenet'=Output    'darknet'=Output

*** Keywords ***
Engine generation
    [Arguments]    ${model}    ${dtype}
    ${cmd}=    catenate    SEPARATOR=
    ...    ${TRTEXEC} --onnx ${TEST_OUT}/${model}_${dtype}/${model}_${dtype}.onnx
    ...    \ --workspace=50
    ...    \ --engine=${TEST_OUT}/${model}_${dtype}/${model}_${dtype}.engine
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    ${rc}    0

Inference with onnx engine and trtexec_json
    [Arguments]    ${model}    ${dtype}
    ${cmd}=    catenate    SEPARATOR=
    ...    ${TRTEXEC_JSON} --engine=${TEST_OUT}/${model}_${dtype}/${model}_${dtype}.engine
    ...    \ --imageDirectoryIn ${TEST_DATA_DIR}/${dtype}
    ...    \ --imageDirectoryOut ${TEST_OUT}/${model}_${dtype}
    ...    \ --verbose
    ...    \ --workspace=168
    ...    \ --writeDetectionVisualizationImage=1
    ...    \ --usePrettyPrintOutput=1
    ...    \ --detectionConfidenceThreshold 0.1
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    ${rc}    0
