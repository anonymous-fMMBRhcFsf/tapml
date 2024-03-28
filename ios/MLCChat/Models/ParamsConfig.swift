//
//  ParamsConfig.swift
//  TapmlChat
//

struct ParamsConfig: Decodable {
    struct ParamsRecord: Decodable {
        let dataPath: String
    }

    let records: [ParamsRecord]
}
