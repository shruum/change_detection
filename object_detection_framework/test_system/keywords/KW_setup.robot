*** Variables ***

${TEST_OUT}    ${CURDIR}/tdo
${TEST_IN}    ${CURDIR}/tdi
${HOME_DIR}    ${CURDIR}/../..

*** Keywords ***

Prepare environment
    [Arguments]    ${data_root}
    Set Environment Variable    VOC_ROOT    ${data_root}
    ${model_file}=    Run    ls -trd ${data_root}/*.pth | tail -1
    Should Not Be Empty    ${model_file}
    Set Suite Variable    ${MODEL_FILE}    ${model_file}
    Create Directory    ${TEST_IN}
    Create Directory    ${TEST_OUT}
