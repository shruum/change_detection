*** Settings ***
Documentation    Facility to test infer_paas_demo.py.

Suite Setup    Suite prepare environment
Suite Teardown    Suite clean environment

Library    OperatingSystem
Library    String

*** Variables ***
${HOME_DIR}    ${CURDIR}/../..
${REPO_BASE}    ${HOME_DIR}/..
@{JSON_FIELDS}    cv_task    obj_num    objects    f_name    f_code    obj_points    x    y    w    h    f_conf
${MODEL_CKPT}    ${REPO_BASE}/models_checkpoints/zoo/det/ssd/resnet18/voc/20.03/model_final.pth
${MODEL_CONFIG}    ${REPO_BASE}/models_checkpoints/zoo/det/ssd/resnet18/voc/20.03/resnet18_ssd512_voc0712.yaml
${MODEL_DIR}    ${HOME_DIR}/model
${SUT}    ${HOME_DIR}/sdk/src/infer_paas_demo.py
${TEST_OUT}    ${HOME_DIR}/output

*** Test Cases ***
Check presence of required fields in output
    [Tags]    infer
    ${rc}    ${log}    Run And Return Rc And Output    ${SUT}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    :For    ${field}    IN    @{JSON_FIELDS}
    \    Should Contain    ${log}    ${field}


*** Keywords ***
Suite prepare environment
    Suite clean environment
    Create Directory    ${MODEL_DIR}
    Run    cp ${MODEL_CONFIG} ${MODEL_DIR}/config.yaml
    Run    cp ${MODEL_CKPT} ${MODEL_DIR}/checkpoints.pth

Suite clean environment
    Remove Directory    ${MODEL_DIR}    recursive=True
    Remove Directory    ${TEST_OUT}    recursive=True
