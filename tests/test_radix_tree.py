from tllm.commons.radix_tree import RadixTree

if __name__ == "__main__":
    tree = RadixTree()
    tree.append_to_request([151646, 151646, 151644, 9707, 11, 1246, 525, 498, 30, 151645], "123")
    tree.append_to_request([151648], "123")
    tree.append_to_request([271], "123")
    tree.append_to_request([151649], "123")
    tree.append_to_request([271], "123")
    tree.append_to_request([9707], "123")
    tree.append_to_request([0], "123")
    tree.append_to_request([358], "123")
    tree.append_to_request([2776], "123")
    tree.append_to_request([1101], "123")
    tree.append_to_request([264], "123")
    tree.append_to_request([4108], "123")
    tree.append_to_request([17847], "123")
    tree.append_to_request([11], "123")
    tree.append_to_request([773], "123")

    input_ids = [
        151646,
        151646,
        151644,
        9707,
        11,
        1246,
        525,
        498,
        30,
        151645,
        9707,
        0,
        358,
        2776,
        1101,
        264,
        4108,
        17847,
        11,
        773,
        358,
        1513,
        944,
        614,
        15650,
        11,
        714,
        358,
        2776,
        1588,
        323,
        5527,
        311,
        1492,
        498,
        448,
        8820,
        498,
        1184,
        13,
        2585,
        525,
        498,
        3730,
        30,
        26525,
        232,
        151643,
        151644,
        100644,
        104307,
        104472,
        11319,
        151645,
    ]
    longest = tree.longest_common_prefix(input_ids)
    print("longest common prefix:", longest)
    print("hit input ids", input_ids[: longest[1]])

    # longest = tree.longest_common_prefix([1, 2, 3, 4, 6, 7, 8, 9])
    # print("longest common prefix:", longest)

    # longest = tree.longest_common_prefix([1, 2, 3, 4, 6, 7, 8, 9])
    # print("longest common prefix:", longest)

    # longest = tree.longest_common_prefix([1, 2, 3])
    # print("longest common prefix:", longest)
    tree.remove(tree.request_id_map["123"].path)
    longest = tree.longest_common_prefix([1, 2, 3, 4])
    print("longest common prefix:", longest)
