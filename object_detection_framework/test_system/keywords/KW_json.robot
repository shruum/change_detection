*** Settings ***
Library    OperatingSystem

*** Keywords ***

Check detections in json
    [Arguments]    ${id}    ${expected_categories}    ${results_location}
    ${results}=    Get File    ${results_location}/${id}_detections.json
    ${json}=    Evaluate    json.loads('''${results}''')    json
    Log    ${json}
    ${expected_nr_boxes}=    Get Length    ${expected_categories}
    ${data_img}=    Set variable    ${json['imgs']}[${id}]
    ${objects_present}=    Evaluate    'objects' in ${data_img}
    Return From Keyword If    not ${objects_present} and ${expected_nr_boxes} == 0    0
    ${msg}=    catenate    SEPARATOR=\n
    ...    Error: Condition is violated that objects should be present and expected nr of boxes be greater than zero
    ...    expected_nr_boxes=${expected_nr_boxes}, objects present=${objects_present}
    Should Be True    ${objects_present} and ${expected_nr_boxes} > 0    msg=${msg}
    ${data_length}    Get length    ${data_img['objects']}
    Should Be True    ${expected_nr_boxes} == ${data_length}
    @{box_list}=    Set Variable    ${json['imgs']['${id}']['objects']}
    Log    ${box_list}
    Expected number of boxes    ${box_list}    expected=${expected_nr_boxes}
    Expected categories    ${box_list}    expected=${expected_categories}

Expected number of boxes
    [Arguments]    ${object_list}    ${expected}
    ${detected}=    Evaluate    len(${object_list})
    Should Be Equal As Integers    ${detected}    ${expected}

Expected categories
    [Arguments]    ${object_list}    ${expected}
    Log    ${expected}
    ${data_length}    Get Length    ${object_list}
    :For    ${index}    IN RANGE    ${data_length}
    \    ${category}=    Evaluate   ${object_list}[${index}][category]
    \    Log    ${category}
    \    Should Be Equal As Integers    ${category}    ${expected}[${index}]

Run engine with trtexec_json
    [Arguments]    ${executor}       ${model}    ${threshold}    ${test_out}    ${test_data}
    ${cmd}=    catenate    SEPARATOR=
    ...    ${executor}
    ...    \ --engine ${test_out}/${model}.engine
    ...    \ --imageDirectoryIn=${test_data}
    ...    \ --imageDirectoryOut=${test_out}/detections
    ...    \ --writeDetectionVisualizationImage=1
    ...    \ --usePrettyPrintOutput=1
    ...    \ --detectionConfidenceThreshold ${threshold}
    ${rc}    ${log}    Run And Return Rc And Output    ${cmd}
    [Return]    ${rc}    ${log}

