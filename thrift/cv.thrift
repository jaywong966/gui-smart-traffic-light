struct LocationInfo{
    1: map<string,string> encoded_images,
    2: i32 n_pedestrians,
    3: i32 n_vehicles,
    4: list<string> traffic_signals,
    5: i32 count_down
}

struct SnapShots{
    1: string location
    2: map<string,string> encoded_images,
}

service cvService{
    list<string> get_location_list()
    SnapShots get_snapshots(1:string location)
    LocationInfo get_location_info(1:string location)
}