*** Settings ***
Documentation
...    Facility to test infer.py.
...    The validity of the resulting files are covered by unit tests in sdk/test_not_used_yet/engine_test.py.

Suite Setup    Suite prepare environment
Suite Teardown    Suite clean environment
Test Setup    Test prepare environment

Library    Collections
Library    OperatingSystem
Library    String

*** Variables ***
${DEMO_IMG_NAME}    003123.jpg
${DEMO_IMG_FULL_NAME}    ${HOME_DIR}/demo/ssd/${DEMO_IMG_NAME}
${HOME_DIR}    ${CURDIR}/../..
${REPO_BASE}    ${HOME_DIR}/..
${MODEL_COCO_CKPT}    ${REPO_BASE}/models_checkpoints/zoo/det/ssd/resnet18/coco/20.03/model_final.pth
${MODEL_VOC_CKPT}    ${REPO_BASE}/models_checkpoints/zoo/det/ssd/resnet18/voc/20.03/model_final.pth
${MODEL_COCO_CONFIG}    ${REPO_BASE}/models_checkpoints/zoo/det/ssd/resnet18/coco/20.03/resnet18_ssd512_coco.yaml
${MODEL_VOC_CONFIG}    ${REPO_BASE}/models_checkpoints/zoo/det/ssd/resnet18/voc/20.03/resnet18_ssd512_voc0712.yaml
${MODEL_DIR}    ${HOME_DIR}/model
${SUT}    ${HOME_DIR}/infer.py
${TEST_IN}    ${HOME_DIR}/input
${TEST_IN_MORE}    ${HOME_DIR}/demo/ssd
${TEST_OUT}    ${HOME_DIR}/output
${TEST_OUT_CUSTOM}    ${HOME_DIR}/output_custom

*** Test Cases ***
Run with default arguments
    [Tags]    infer
    ${rc}    ${log}    Run And Return Rc And Output    ${SUT}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    File Should Exist    ${TEST_OUT}/${DEMO_IMG_NAME}

Run for invalid dataset
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --dataset_type=invalid
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    2    ${rc}    msg=returned unexpected error code ${rc}
    Should Contain    ${log}    infer.py: error: argument --dataset_type: invalid choice: 'invalid'

Run for COCO dataset
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --ckpt=${MODEL_COCO_CKPT}
    ...    --config-file=${MODEL_COCO_CONFIG}
    ...    --dataset_type=coco
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    File Should Exist    ${TEST_OUT}/${DEMO_IMG_NAME}

Run for VOC dataset
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --ckpt=${MODEL_VOC_CKPT}
    ...    --config-file=${MODEL_VOC_CONFIG}
    ...    --dataset_type=voc
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    File Should Exist    ${TEST_OUT}/${DEMO_IMG_NAME}

Output format img
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --output_format=img
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    File Should Exist    ${TEST_OUT}/${DEMO_IMG_NAME}

Output format json
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --output_format=json
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    ${basename}    ${ext}=    Split Extension    ${DEMO_IMG_NAME}
    File Should Exist    ${TEST_OUT}/${basename}.json

Output format json_nie
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --output_format=json_nie
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    ${basename}    ${ext}=    Split Extension    ${DEMO_IMG_NAME}
    File Should Exist    ${TEST_OUT}/${basename}.json

Output format txt
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --output_format=txt
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    ${basename}    ${ext}=    Split Extension    ${DEMO_IMG_NAME}
    File Should Exist    ${TEST_OUT}/${basename}.txt

Output format xml
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --output_format=xml
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    ${basename}    ${ext}=    Split Extension    ${DEMO_IMG_NAME}
    File Should Exist    ${TEST_OUT}/${basename}.xml

Output format unsupported
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --output_format=unsupported
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    2    ${rc}    msg=returned unexpected error code ${rc}
    Should Contain    ${log}    infer.py: error: argument --output_format: invalid choice

Inexistend input file
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --input=inexistent.jpg
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    Should Contain    ${log}    Warning: inexistent.jpg does not exist!

Run on multiple images
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --input=${TEST_IN_MORE}
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    ${input_files}=    List Files In Directory    ${TEST_IN_MORE}
    ${output_files}=    List Files In Directory    ${TEST_OUT}
    Lists Should Be Equal    ${input_files}    ${output_files}

Use custom output directory
    [Tags]    infer
    [Teardown]    Remove Directory    ${TEST_OUT_CUSTOM}    recursive=True
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --output_dir=${TEST_OUT_CUSTOM}
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    ${input_files}=    List Files In Directory    ${TEST_IN}
    ${output_files}=    List Files In Directory    ${TEST_OUT_CUSTOM}
    Lists Should Be Equal    ${input_files}    ${output_files}

Use threshould out of range
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --score_threshold=2
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    2    ${rc}    msg=returned unexpected error code ${rc}
    Should Contain    ${log}    infer.py: error: argument --score_threshold: 2.0 not in range [0.0, 1.0]

Use threshold max value
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --score_threshold=1
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    File Should Exist    ${TEST_OUT}/${DEMO_IMG_NAME}

Use threshold min value
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --score_threshold=0.0
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    File Should Exist    ${TEST_OUT}/${DEMO_IMG_NAME}

Use threshold incorrect type value
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --score_threshold=string
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    2    ${rc}    msg=returned unexpected error code ${rc}
    Should Contain    ${log}    infer.py: error: argument --score_threshold: string not a floating-point literal

Display help
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --help
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    Should Contain    ${log}    Inference application.

Use verbose
    [Tags]    infer
    ${cmd}=    catenate    SEPARATOR=${SPACE}
    ...    ${SUT}
    ...    --verbose
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    Log    ${log}
    Should Be Equal As Integers    0    ${rc}    msg=python returned error code ${rc}
    Should Contain    ${log}    Run inference on


*** Keywords ***
Suite prepare environment
    Suite clean environment
    Create Directory    ${MODEL_DIR}
    Create Directory    ${TEST_IN}
    Run    cp ${MODEL_VOC_CONFIG} ${MODEL_DIR}/config.yaml
    Run    cp ${MODEL_VOC_CKPT} ${MODEL_DIR}/checkpoints.pth
    Run    cp ${DEMO_IMG_FULL_NAME} ${TEST_IN}

Suite clean environment
    Remove Directory    ${MODEL_DIR}    recursive=True
    Remove Directory    ${TEST_IN}    recursive=True
    Remove Directory    ${TEST_OUT}    recursive=True

Test prepare environment
    Remove Directory    ${TEST_OUT}    recursive=True
    Create Directory    ${TEST_OUT}


