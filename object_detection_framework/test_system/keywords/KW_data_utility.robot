*** Settings ***
Library    OperatingSystem
Library    String

*** Variables ***
${TEST_DATA_DIR}    /data/nie/teams/arl/system_tests_data/object_detection_framework/test_data
${TEST_OUT}    ${CURDIR}/tdo

*** Keywords ***
Check result
    [Arguments]    ${model}    ${dtype}
    @{files}    List Files In Directory       ${TEST_DATA_DIR}/${dtype}
    :FOR    ${file}    IN     @{files}
    \    ${basename}    Extract file basename    ${file}
    \    Check json content    ${model}_${dtype}    ${basename}

Extract file basename
    [Arguments]    ${file}
    ${path}    ${ext}    Split Extension    ${file}
    ${path}    ${basename}    Split Path    ${path}
    [Return]    ${basename}

Check json content
    [Arguments]    ${mode}    ${basename}
    File Should Exist   ${TEST_OUT}/${mode}/${basename}_detections.json
    Compare json    ${TEST_OUT}/${mode}/${basename}_detections.json    ${TEST_DATA_DIR}/${mode}/${basename}_detections.json    ${basename}

Compare json
    [Arguments]    ${data_file}    ${ref_file}    ${basename}
    ${input_string}    Get File    ${data_file}
    ${ref_string}    Get File    ${ref_file}
    ${data}    Evaluate    json.loads('''${input_string}''')    json
    ${ref}    Evaluate    json.loads('''${ref_string}''')    json
    ${data_img}    Set variable    ${data['imgs']}[${basename}]
    ${ref_img}    Set variable    ${ref['imgs']}[${basename}]
    Should Be Equal As Integers    ${data_img['height']}    ${ref_img['height']}
    Should Be Equal As Integers    ${data_img['width']}    ${ref_img['width']}
    ${data_length}    Get length    ${data_img['objects']}
    ${ref_length}    Get length    ${ref_img['objects']}
    Should Be Equal As Integers    ${data_length}    ${ref_length}
    :For    ${index}    IN RANGE    ${data_length}
    \    ${data_box}    Set variable    ${data_img['objects']}[${index}]
    \    ${ref_box}    Set variable    ${ref_img['objects']}[${index}]
    \    Log    data xmax: ${data_box['bbox']['xmax']}
    \    Log    ref xmax: ${ref_box['bbox']['xmax']}
    \    Run Keyword And Continue On Failure    Should Be Equal As Integers    ${data_box['bbox']['xmax']}    ${ref_box['bbox']['xmax']}
    \    Run Keyword And Continue On Failure    Should Be Equal As Integers    ${data_box['bbox']['xmin']}    ${ref_box['bbox']['xmin']}
    \    Run Keyword And Continue On Failure    Should Be Equal As Integers    ${data_box['bbox']['ymax']}    ${ref_box['bbox']['ymax']}
    \    Run Keyword And Continue On Failure    Should Be Equal As Integers    ${data_box['bbox']['ymin']}    ${ref_box['bbox']['ymin']}
    \    Run Keyword And Continue On Failure    Should Be Equal    ${data_box['category']}    ${ref_box['category']}
    \    Run Keyword And Continue On Failure    Should Be Equal    ${data_box['color']}    ${ref_box['color']}
    \    Run Keyword And Continue On Failure    Should Be Equal    ${data_box['contourl']}    ${ref_box['contourl']}
    \    Run Keyword And Continue On Failure    Should Be Equal    ${data_box['polygon']}    ${ref_box['polygon']}
    \    Run Keyword And Continue On Failure    Should Be Equal    ${data_box['score']}    ${ref_box['score']}
