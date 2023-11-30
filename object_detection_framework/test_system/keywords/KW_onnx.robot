*** Settings ***
Library    OperatingSystem
Library    String

*** Variables ***
${CONFIG_DIR}    /data/nie/teams/arl/system_tests_data/object_detection_framework/ssd
${TEST_OUT}    ${CURDIR}/tdo
${HOME_DIR}    ${CURDIR}/../..

*** Keywords ***
Convert PyTorch model into ONNX ssd
    [Arguments]    ${model}    ${dtype}
    ${config_file}=    Run    ls ${CONFIG_DIR}/${model}/20.01/*${dtype}*.yaml
    ${cmd}=    catenate    SEPARATOR=
    ...    ${HOME_DIR}/modeltoonnx.py
    ...    \ --config_file=${config_file}
    ...    \ --export_path=${TEST_OUT}/${model}_${dtype}/${model}_${dtype}.onnx
    ...    \ --pretrained=True
    ...    \ --pretrained_path=${CONFIG_DIR}/${model}/20.01/${model}_${dtype}.pth
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc} for model ${model} datatype ${dtype}    values=False

Generated onnx model should be valid
    [Arguments]    ${model}    ${backbone_nr}    ${header_line_nr}    ${nms_nr}
    ${rc}    ${log}    Run And Return Rc And Output    python3 -c "import onnx; model = onnx.load('${TEST_OUT}/${model}'); print(onnx.helper.printable_graph(model.graph))" > ${TEST_OUT}/${model}.graph
    Log    ${log}
    Run Keyword And Continue On Failure    Should be ${backbone_nr} backbone lines generated for ${TEST_OUT}/${model}.graph
    Run Keyword And Continue On Failure    Should be ${header_line_nr} yolo header lines generated for ${TEST_OUT}/${model}.graph
    Run Keyword And Continue On Failure    Should be ${nms_nr} nms lines generated for ${TEST_OUT}/${model}.graph

Should be ${expected} backbone lines generated for ${model}
    ${lines}=    Run    grep -c "^ *%backbone" ${model}
    Should Be Equal As Integers    ${expected}    ${lines}

Should be ${expected} yolo header lines generated for ${model}
    ${lines}=    Run    grep -c "^ *%head" ${model}
    Should Be Equal As Integers    ${expected}    ${lines}

Should be ${expected} nms lines generated for ${model}
    ${lines}=    Run    grep -c "nms_TRT" ${model}
    Should Be Equal As Integers    ${expected}    ${lines}

