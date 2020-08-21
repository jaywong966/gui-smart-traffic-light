// renderer.js
var thrift = require('thrift');
var cvService = require('./gen-nodejs/cvService.js');
var ttypes = require('./gen-nodejs/cv_types.js');
var thriftConnection = thrift.createConnection('127.0.0.1', 9999);
var thriftClient = thrift.createClient(cvService, thriftConnection);

thriftConnection.on("error",function(e)
{
    console.log(e);
});

var targetLocation = "";
var map_targetLocation = "";
var liveFrameUpdatePeriod = 100; //100ms




$('#tab1-locations').load('tab1-locations.html', function() {
    $('#btn-refresh-locations').on("click", function(e){
        refreshLocationList();
    })

    $('#location-list').on("click", function(e){
        var targetElement = $(e.target);
        if (targetElement.attr('class') == 'caption'){
            var location = targetElement.text();
            targetLocation = location;
        }
    })

    function clearLocationList(lv){
        var groups = lv.find('.node-group');
        if (groups.length > 0){
            for(var i=0; i<groups.length; i++){
                lv.data('listview').del(groups[i])
            }
            
        }
    }

    function addSnapshotsToLocationList(lv, locationName, snapshots){
        lv.data('listview').addGroup({caption: locationName});
        var groups = lv.find('.node-group');
        var parent = $(groups[groups.length-1]);
        for(var name in snapshots){
            var imgSrc = "data:image/jpg;base64,"+snapshots[name];
            lv.data('listview').add(parent, {
                caption: name,
                icon: '<img src="'+imgSrc+'">'
            })
        }
    }

    function refreshLocationList(){
        var lv = $('#location-list');
        clearLocationList(lv);

        thriftClient.get_location_list((error, result) => {
            if(error) {
                console.error(error);
            } 
            else 
            {
                var locations = result;
                locations.forEach(location=>{
                    target_location = location;
                    thriftClient.get_snapshots(target_location, (error, result) => {
                        if(error) {
                            console.error(error);
                        } 
                        else 
                        {
                            // console.log(result)
                            var location = result.location;
                            var snapshots = result.encoded_images;
                            addSnapshotsToLocationList(lv, location, snapshots);
                        }
                    });
                })
            }
        });
    }


});

$("#tab1-videos").load("tab1-videos.html", function() {
    var image_json = {
        "template": "<figure class='text-center'><div class='img-container thumbnail'><img src='<%this.img_src%>'></div><figcaption class='text-bold'><%this.caption%></figcaption></figure>",
        "data": []
    }
    setInterval(function(){
        if (targetLocation != ""){
            thriftClient.get_location_info(targetLocation, (error, result) => {
                if(error) {
                    console.error(error);
                }
                else
                {
                    images = result.encoded_images
                    image_json['data'] = [];
                    camera_id = 1
                    for (var key in images) {
                        image  = images[key];
                        imgSrc = "data:image/jpg;base64,"+image;
                        image_json['data'].push(
                            {
                                "img_src": imgSrc,
                                "caption": key
                            }
                        )
                        // document.querySelector('#camera'+camera_id).setAttribute("src", imgSrc);
                        camera_id += 1;
                    }
                    document.querySelector('#pedestrians').innerHTML = result.n_pedestrians;
                    document.querySelector('#vehicles').innerHTML = result.n_vehicles;
                    document.querySelector('#traffic-light-count-down').innerHTML = result.count_down;
                }
            });

            if (image_json['data'].length > 0){
                var list = $('#list-videos').data('list');
                list._createItemsFromJSON(image_json);
                list.draw();
            }
        }
    }, liveFrameUpdatePeriod);
});

$("#dashboard").load("dashboard.html", function() {
    var map_image_json = {
        "template": "<figure class='text-center'><div class='img-container thumbnail'><img src='<%this.img_src%>'></div><figcaption class='text-bold'><%this.caption%></figcaption></figure>",
        "data": []
    }

    setInterval(function(){
        map_targetLocation = get_location();
        if (map_targetLocation != ""){
            thriftClient.get_location_info(map_targetLocation, (error, result) => {
                if(error) {
                    console.error(error);
                }
                else
                {
                    images = result.encoded_images
                    map_image_json['data'] = [];
                    camera_id = 1
                    for (var key in images) {
                        image  = images[key];
                        imgSrc = "data:image/jpg;base64,"+image;
                        map_image_json['data'].push(
                            {
                                "img_src": imgSrc,
                                "caption": key
                            }
                        )
                        // document.querySelector('#camera'+camera_id).setAttribute("src", imgSrc);
                        camera_id += 1;
                    }

                    document.querySelector('#map-pedestrians').innerHTML = result.n_pedestrians;
                    document.querySelector('#map-vehicles').innerHTML = result.n_vehicles;
                    document.querySelector('#map-traffic-light-count-down').innerHTML = result.count_down;

                }
            });

            if (map_image_json['data'].length > 0){
                var list = $('#map-list-videos').data('list');
                list._createItemsFromJSON(map_image_json);
                list.draw();
            }
        }
    }, liveFrameUpdatePeriod);
});
