def output_template(kinesis_event, new_video, message_receive_time):
    default_time = "1970-01-01T00:00:00Z"
    # general outputs
    output_dict = {}
    output_dict["LOMOTIF_ID"] = kinesis_event["lomotif"]["id"]
    output_dict["VIDEO"] = new_video
    output_dict["COUNTRY"] = kinesis_event["lomotif"]["country"]
    output_dict["CREATION_TIME"] = kinesis_event["lomotif"]["created"]
    output_dict["MESSAGE_RECEIVE_TIME"] = message_receive_time
    output_dict["KEY_FRAMES"] = ""
    output_dict["NUM_FRAMES"] = -1
    output_dict["FPS"] = -1

    # caption model outputs
    output_dict["CAPTION_PROCESS_START_TIME"] = default_time
    output_dict["CAPTION_PREDICTION_TIME"] = default_time
    output_dict["CAPTION_STATUS"] = -1

    # embedding model outputs
    output_dict["EMBED_PROCESS_START_TIME"] = default_time
    output_dict["EMBED_PREDICTION_TIME"] = default_time
    output_dict["EMBED_STATUS"] = -1

    return output_dict
