SYSTEM_PROMPT = "When given a video and a query, call the relevant function only once with the appropriate timecodes and text for the video"

USER_PROMPT = 'Generate chart data for this video based on the following instructions: for each scene, count the number of people visible. Call set_timecodes_with_numeric_values once with the list of data values and timecodes.'

GRAPHICS_USER_PROMPT = 'Generate chart data for this video based on the following instructions: for each scene, count all logos, motion graphics, titles/lower-thirds visible-- for every second of the video give accurate integer. Call set_timecodes_with_numeric_values once with the list of data values and timecodes.'