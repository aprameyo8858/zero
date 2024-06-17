import json

flower102_maps = {
    "21": "fire lily", 
    "3": "canterbury bells", 
    "45": "bolero deep blue",
    "1": "pink primrose",
    "34": "mexican aster",
    "27": "prince of wales feathers",
    "7": "moon orchid",
    "16": "globe-flower",
    "25": "grape hyacinth",
    "26": "corn poppy",
    "79": "toad lily",
    "39": "siam tulip",
    "24": "red ginger",
    "67": "spring crocus",
    "35": "alpine sea holly",
    "32": "garden phlox",
    "10": "globe thistle",
    "6": "tiger lily",
    "93": "ball moss",
    "33": "love in the mist",
    "9": "monkshood",
    "102": "blackberry lily",
    "14": "spear thistle",
    "19": "balloon flower",
    "100": "blanket flower",
    "13": "king protea",
    "49": "oxeye daisy",
    "15": "yellow iris",
    "61": "cautleya spicata",
    "31": "carnation",
    "64": "silverbush",
    "68": "bearded iris",
    "63": "black-eyed susan",
    "69": "windflower",
    "62": "japanese anemone",
    "20": "giant white arum lily",
    "38": "great masterwort",
    "4": "sweet pea", 
    "86": "tree mallow", 
    "101": "trumpet creeper", 
    "42": "daffodil", 
    "22": "pincushion flower", 
    "2": "hard-leaved pocket orchid", 
    "54": "sunflower", 
    "66": "osteospermum", 
    "70": "tree poppy", 
    "85": "desert-rose", 
    "99": "bromelia",
    "87": "magnolia", 
    "5": "english marigold", 
    "92": "bee balm", 
    "28": "stemless gentian", 
    "97": "mallow", 
    "57": "gaura", 
    "40": "lenten rose", 
    "47": "marigold", 
    "59": "orange dahlia", 
    "48": "buttercup", 
    "55": "pelargonium", 
    "36": "ruby-lipped cattleya", 
    "91": "hippeastrum", 
    "29": "artichoke", 
    "71": "gazania", 
    "90": "canna lily", 
    "18": "peruvian lily", 
    "98": "mexican petunia", 
    "8": "bird of paradise", 
    "30": "sweet william", 
    "17": "purple coneflower", 
    "52": "wild pansy", 
    "84": "columbine", 
    "12": "colt's foot", 
    "11": "snapdragon", 
    "96": "camellia", 
    "23": "fritillary", 
    "50": "common dandelion", 
    "44": "poinsettia", 
    "53": "primula", 
    "72": "azalea", 
    "65": "californian poppy", 
    "80": "anthurium", 
    "76": "morning glory", 
    "37": "cape flower", 
    "56": "bishop of llandaff", 
    "60": "pink-yellow dahlia", 
    "82": "clematis", 
    "58": "geranium", 
    "75": "thorn apple", 
    "41": "barbeton daisy", 
    "95": "bougainvillea", 
    "43": "sword lily", 
    "83": "hibiscus", 
    "78": "lotus", 
    "88": "cyclamen", 
    "94": "foxglove", 
    "81": "frangipani", 
    "74": "rose", 
    "89": "watercress", 
    "73": "water lily", 
    "46": "wallflower", 
    "77": "passion flower", 
    "51": "petunia"
}

flower102_maps = {int(k)-1: v for k, v in flower102_maps.items()}
flower102_maps = dict(sorted(flower102_maps.items()))
flower102_classes = list(flower102_maps.values())

# list of classnames for food101 from the TPT repository
# food101_classes = ['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito','bruschetta','caesar_salad','cannoli','caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi',
#                    'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

# from the MaPLe repository
food101_classes = ['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito','bruschetta','caesar_salad','cannoli','caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi',
                   'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

dtd_classes = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined',
               'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']

pets_classes = ['abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'bengal', 'birman', 'bombay', 'boxer', 'british_shorthair', 'chihuahua', 'egyptian_mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
                'keeshond', 'leonberger', 'maine_coon', 'miniature_pinscher', 'newfoundland', 'persian', 'pomeranian', 'pug', 'ragdoll', 'russian_blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'siamese', 'sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

sun397_classes = ['abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'outdoor apartment_building', 'indoor apse', 'aquarium', 'aqueduct', 'arch', 'archive', 'outdoor arrival_gate', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'outdoor athletic_field', 'public atrium', 'attic', 'auditorium', 'auto_factory', 'badlands', 'indoor badminton_court', 'baggage_claim', 'shop bakery', 'exterior balcony', 'interior balcony', 'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basilica', 'outdoor basketball_court', 'bathroom', 'batters_box', 'bayou', 'indoor bazaar', 'outdoor bazaar', 'beach', 'beauty_salon', 'bedroom', 'berth', 'biology_laboratory', 'indoor bistro', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'indoor booth', 'botanical_garden', 'indoor bow_window', 'outdoor bow_window', 'bowling_alley', 'boxing_ring', 'indoor brewery', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'outdoor cabin', 'cafeteria', 'campsite', 'campus', 'natural canal', 'urban canal', 'candy_store', 'canyon', 'backseat car_interior', 'frontseat car_interior', 'carrousel', 'indoor casino', 'castle', 'catacomb', 'indoor cathedral', 'outdoor cathedral', 'indoor cavern', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab', 'indoor chicken_coop', 'outdoor chicken_coop', 'childs_room', 'indoor church', 'outdoor church', 'classroom', 'clean_room', 'cliff', 'indoor cloister', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'control_room', 'outdoor control_tower', 'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse', 'courtroom', 'courtyard', 'exterior covered_bridge', 'creek', 'crevasse', 'crosswalk', 'office cubicle', 'dam', 'delicatessen', 'dentists_office', 'sand desert', 'vegetation desert', 'indoor diner', 'outdoor diner', 'home dinette', 'vehicle dinette', 'dining_car', 'dining_room', 'discotheque', 'dock', 'outdoor doorway', 'dorm_room', 'driveway', 'outdoor driving_range', 'drugstore', 'electrical_substation', 'door elevator', 'interior elevator', 'elevator_shaft', 'engine_room', 'indoor escalator', 'excavation', 'indoor factory', 'fairway', 'fastfood_restaurant', 'cultivated field', 'wild field', 'fire_escape', 'fire_station', 'indoor firing_range', 'fishpond', 'indoor florist_shop', 'food_court', 'broadleaf forest', 'needleleaf forest', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'game_room', 'indoor garage', 'garbage_dump', 'gas_station', 'exterior gazebo', 'indoor general_store', 'outdoor general_store', 'gift_shop', 'golf_course', 'indoor greenhouse', 'outdoor greenhouse', 'indoor gymnasium', 'indoor hangar', 'outdoor hangar', 'harbor', 'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office', 'hospital', 'hospital_room', 'hot_spring', 'outdoor hot_tub', 'outdoor hotel', 'hotel_room', 'house', 'outdoor hunting_lodge', 'ice_cream_parlor', 'ice_floe', 'ice_shelf',
                  'indoor ice_skating_rink', 'outdoor ice_skating_rink', 'iceberg', 'igloo', 'industrial_area', 'outdoor inn', 'islet', 'indoor jacuzzi', 'indoor jail', 'jail_cell', 'jewelry_shop', 'kasbah', 'indoor kennel', 'outdoor kennel', 'kindergarden_classroom', 'kitchen', 'kitchenette', 'outdoor labyrinth', 'natural lake', 'landfill', 'landing_deck', 'laundromat', 'lecture_room', 'indoor library', 'outdoor library', 'outdoor lido_deck', 'lift_bridge', 'lighthouse', 'limousine_interior', 'living_room', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'indoor market', 'outdoor market', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'water moat', 'outdoor monastery', 'indoor mosque', 'outdoor mosque', 'motel', 'mountain', 'mountain_snowy', 'indoor movie_theater', 'indoor museum', 'music_store', 'music_studio', 'outdoor nuclear_power_plant', 'nursery', 'oast_house', 'outdoor observatory', 'ocean', 'office', 'office_building', 'outdoor oil_refinery', 'oilrig', 'operating_room', 'orchard', 'outdoor outhouse', 'pagoda', 'palace', 'pantry', 'park', 'indoor parking_garage', 'outdoor parking_garage', 'parking_lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'indoor pilothouse', 'outdoor planetarium', 'playground', 'playroom', 'plaza', 'indoor podium', 'outdoor podium', 'pond', 'establishment poolroom', 'home poolroom', 'outdoor power_plant', 'promenade_deck', 'indoor pub', 'pulpit', 'putting_green', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'riding_arena', 'river', 'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea_cliff', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'indoor shopping_mall', 'shower', 'skatepark', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash_court', 'stable', 'baseball stadium', 'football stadium', 'indoor stage', 'staircase', 'street', 'subway_interior', 'platform subway_station', 'supermarket', 'sushi_bar', 'swamp', 'indoor swimming_pool', 'outdoor swimming_pool', 'indoor synagogue', 'outdoor synagogue', 'television_studio', 'east_asia temple', 'south_asia temple', 'indoor tennis_court', 'outdoor tennis_court', 'outdoor tent', 'indoor_procenium theater', 'indoor_seats theater', 'thriftshop', 'throne_room', 'ticket_booth', 'toll_plaza', 'topiary_garden', 'tower', 'toyshop', 'outdoor track', 'train_railway', 'platform train_station', 'tree_farm', 'tree_house', 'trench', 'coral_reef underwater', 'utility_room', 'valley', 'van_interior', 'vegetable_garden', 'veranda', 'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'indoor volleyball_court', 'outdoor volleyball_court', 'waiting_room', 'indoor warehouse', 'water_tower', 'block waterfall', 'fan waterfall', 'plunge waterfall', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'barrel_storage wine_cellar', 'bottle_storage wine_cellar', 'indoor wrestling_ring', 'yard', 'youth_hostel']

caltech101_classes = ['face', 'leopard', 'motorbike', 'accordion', 'airplane', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone',
                      'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']

cars_classes = ['2000 AM General Hummer SUV', '2012 Acura RL Sedan', '2012 Acura TL Sedan', '2008 Acura TL Type-S', '2012 Acura TSX Sedan', '2001 Acura Integra Type R', '2012 Acura ZDX Hatchback', '2012 Aston Martin V8 Vantage Convertible', '2012 Aston Martin V8 Vantage Coupe', '2012 Aston Martin Virage Convertible', '2012 Aston Martin Virage Coupe', '2008 Audi RS 4 Convertible', '2012 Audi A5 Coupe', '2012 Audi TTS Coupe', '2012 Audi R8 Coupe', '1994 Audi V8 Sedan', '1994 Audi 100 Sedan', '1994 Audi 100 Wagon', '2011 Audi TT Hatchback', '2011 Audi S6 Sedan', '2012 Audi S5 Convertible', '2012 Audi S5 Coupe', '2012 Audi S4 Sedan', '2007 Audi S4 Sedan', '2012 Audi TT RS Coupe', '2012 BMW ActiveHybrid 5 Sedan', '2012 BMW 1 Series Convertible', '2012 BMW 1 Series Coupe', '2012 BMW 3 Series Sedan', '2012 BMW 3 Series Wagon', '2007 BMW 6 Series Convertible', '2007 BMW X5 SUV', '2012 BMW X6 SUV', '2012 BMW M3 Coupe', '2010 BMW M5 Sedan', '2010 BMW M6 Convertible', '2012 BMW X3 SUV', '2012 BMW Z4 Convertible', '2012 Bentley Continental Supersports Conv. Convertible', '2009 Bentley Arnage Sedan', '2011 Bentley Mulsanne Sedan', '2012 Bentley Continental GT Coupe', '2007 Bentley Continental GT Coupe', '2007 Bentley Continental Flying Spur Sedan', '2009 Bugatti Veyron 16.4 Convertible', '2009 Bugatti Veyron 16.4 Coupe', '2012 Buick Regal GS', '2007 Buick Rainier SUV', '2012 Buick Verano Sedan', '2012 Buick Enclave SUV', '2012 Cadillac CTS-V Sedan', '2012 Cadillac SRX SUV', '2007 Cadillac Escalade EXT Crew Cab', '2012 Chevrolet Silverado 1500 Hybrid Crew Cab', '2012 Chevrolet Corvette Convertible', '2012 Chevrolet Corvette ZR1', '2007 Chevrolet Corvette Ron Fellows Edition Z06', '2012 Chevrolet Traverse SUV', '2012 Chevrolet Camaro Convertible', '2010 Chevrolet HHR SS', '2007 Chevrolet Impala Sedan', '2012 Chevrolet Tahoe Hybrid SUV', '2012 Chevrolet Sonic Sedan', '2007 Chevrolet Express Cargo Van', '2012 Chevrolet Avalanche Crew Cab', '2010 Chevrolet Cobalt SS', '2010 Chevrolet Malibu Hybrid Sedan', '2009 Chevrolet TrailBlazer SS', '2012 Chevrolet Silverado 2500HD Regular Cab', '2007 Chevrolet Silverado 1500 Classic Extended Cab', '2007 Chevrolet Express Van', '2007 Chevrolet Monte Carlo Coupe', '2007 Chevrolet Malibu Sedan', '2012 Chevrolet Silverado 1500 Extended Cab', '2012 Chevrolet Silverado 1500 Regular Cab', '2009 Chrysler Aspen SUV', '2010 Chrysler Sebring Convertible', '2012 Chrysler Town and Country Minivan', '2010 Chrysler 300 SRT-8', '2008 Chrysler Crossfire Convertible', '2008 Chrysler PT Cruiser Convertible', '2002 Daewoo Nubira Wagon', '2012 Dodge Caliber Wagon', '2007 Dodge Caliber Wagon', '1997 Dodge Caravan Minivan', '2010 Dodge Ram Pickup 3500 Crew Cab', '2009 Dodge Ram Pickup 3500 Quad Cab', '2009 Dodge Sprinter Cargo Van', '2012 Dodge Journey SUV', '2010 Dodge Dakota Crew Cab', '2007 Dodge Dakota Club Cab', '2008 Dodge Magnum Wagon', '2011 Dodge Challenger SRT8', '2012 Dodge Durango SUV', '2007 Dodge Durango SUV', '2012 Dodge Charger Sedan', '2009 Dodge Charger SRT-8',
                '1998 Eagle Talon Hatchback', '2012 FIAT 500 Abarth', '2012 FIAT 500 Convertible', '2012 Ferrari FF Coupe', '2012 Ferrari California Convertible', '2012 Ferrari 458 Italia Convertible', '2012 Ferrari 458 Italia Coupe', '2012 Fisker Karma Sedan', '2012 Ford F-450 Super Duty Crew Cab', '2007 Ford Mustang Convertible', '2007 Ford Freestar Minivan', '2009 Ford Expedition EL SUV', '2012 Ford Edge SUV', '2011 Ford Ranger SuperCab', '2006 Ford GT Coupe', '2012 Ford F-150 Regular Cab', '2007 Ford F-150 Regular Cab', '2007 Ford Focus Sedan', '2012 Ford E-Series Wagon Van', '2012 Ford Fiesta Sedan', '2012 GMC Terrain SUV', '2012 GMC Savana Van', '2012 GMC Yukon Hybrid SUV', '2012 GMC Acadia SUV', '2012 GMC Canyon Extended Cab', '1993 Geo Metro Convertible', '2010 HUMMER H3T Crew Cab', '2009 HUMMER H2 SUT Crew Cab', '2012 Honda Odyssey Minivan', '2007 Honda Odyssey Minivan', '2012 Honda Accord Coupe', '2012 Honda Accord Sedan', '2012 Hyundai Veloster Hatchback', '2012 Hyundai Santa Fe SUV', '2012 Hyundai Tucson SUV', '2012 Hyundai Veracruz SUV', '2012 Hyundai Sonata Hybrid Sedan', '2007 Hyundai Elantra Sedan', '2012 Hyundai Accent Sedan', '2012 Hyundai Genesis Sedan', '2012 Hyundai Sonata Sedan', '2012 Hyundai Elantra Touring Hatchback', '2012 Hyundai Azera Sedan', '2012 Infiniti G Coupe IPL', '2011 Infiniti QX56 SUV', '2008 Isuzu Ascender SUV', '2012 Jaguar XK XKR', '2012 Jeep Patriot SUV', '2012 Jeep Wrangler SUV', '2012 Jeep Liberty SUV', '2012 Jeep Grand Cherokee SUV', '2012 Jeep Compass SUV', '2008 Lamborghini Reventon Coupe', '2012 Lamborghini Aventador Coupe', '2012 Lamborghini Gallardo LP 570-4 Superleggera', '2001 Lamborghini Diablo Coupe', '2012 Land Rover Range Rover SUV', '2012 Land Rover LR2 SUV', '2011 Lincoln Town Car Sedan', '2012 MINI Cooper Roadster Convertible', '2012 Maybach Landaulet Convertible', '2011 Mazda Tribute SUV', '2012 McLaren MP4-12C Coupe', '1993 Mercedes-Benz 300-Class Convertible', '2012 Mercedes-Benz C-Class Sedan', '2009 Mercedes-Benz SL-Class Coupe', '2012 Mercedes-Benz E-Class Sedan', '2012 Mercedes-Benz S-Class Sedan', '2012 Mercedes-Benz Sprinter Van', '2012 Mitsubishi Lancer Sedan', '2012 Nissan Leaf Hatchback', '2012 Nissan NV Passenger Van', '2012 Nissan Juke Hatchback', '1998 Nissan 240SX Coupe', '1999 Plymouth Neon Coupe', '2012 Porsche Panamera Sedan', '2012 Ram C/V Cargo Van Minivan', '2012 Rolls-Royce Phantom Drophead Coupe Convertible', '2012 Rolls-Royce Ghost Sedan', '2012 Rolls-Royce Phantom Sedan', '2012 Scion xD Hatchback', '2009 Spyker C8 Convertible', '2009 Spyker C8 Coupe', '2007 Suzuki Aerio Sedan', '2012 Suzuki Kizashi Sedan', '2012 Suzuki SX4 Hatchback', '2012 Suzuki SX4 Sedan', '2012 Tesla Model S Sedan', '2012 Toyota Sequoia SUV', '2012 Toyota Camry Sedan', '2012 Toyota Corolla Sedan', '2012 Toyota 4Runner SUV', '2012 Volkswagen Golf Hatchback', '1991 Volkswagen Golf Hatchback', '2012 Volkswagen Beetle Hatchback', '2012 Volvo C30 Hatchback', '1993 Volvo 240 Sedan', '2007 Volvo XC90 SUV', '2012 smart fortwo Convertible']

ucf101_classes = ['Apply_Eye_Makeup', 'Apply_Lipstick', 'Archery', 'Baby_Crawling', 'Balance_Beam', 'Band_Marching', 'Baseball_Pitch', 'Basketball', 'Basketball_Dunk', 'Bench_Press', 'Biking', 'Billiards', 'Blow_Dry_Hair', 'Blowing_Candles', 'Body_Weight_Squats', 'Bowling', 'Boxing_Punching_Bag', 'Boxing_Speed_Bag', 'Breast_Stroke', 'Brushing_Teeth', 'Clean_And_Jerk', 'Cliff_Diving', 'Cricket_Bowling', 'Cricket_Shot', 'Cutting_In_Kitchen', 'Diving', 'Drumming', 'Fencing', 'Field_Hockey_Penalty', 'Floor_Gymnastics', 'Frisbee_Catch', 'Front_Crawl', 'Golf_Swing', 'Haircut', 'Hammering', 'Hammer_Throw', 'Handstand_Pushups', 'Handstand_Walking', 'Head_Massage', 'High_Jump', 'Horse_Race', 'Horse_Riding', 'Hula_Hoop', 'Ice_Dancing', 'Javelin_Throw', 'Juggling_Balls', 'Jumping_Jack', 'Jump_Rope',
                  'Kayaking', 'Knitting', 'Long_Jump', 'Lunges', 'Military_Parade', 'Mixing', 'Mopping_Floor', 'Nunchucks', 'Parallel_Bars', 'Pizza_Tossing', 'Playing_Cello', 'Playing_Daf', 'Playing_Dhol', 'Playing_Flute', 'Playing_Guitar', 'Playing_Piano', 'Playing_Sitar', 'Playing_Tabla', 'Playing_Violin', 'Pole_Vault', 'Pommel_Horse', 'Pull_Ups', 'Punch', 'Push_Ups', 'Rafting', 'Rock_Climbing_Indoor', 'Rope_Climbing', 'Rowing', 'Salsa_Spin', 'Shaving_Beard', 'Shotput', 'Skate_Boarding', 'Skiing', 'Skijet', 'Sky_Diving', 'Soccer_Juggling', 'Soccer_Penalty', 'Still_Rings', 'Sumo_Wrestling', 'Surfing', 'Swing', 'Table_Tennis_Shot', 'Tai_Chi', 'Tennis_Swing', 'Throw_Discus', 'Trampoline_Jumping', 'Typing', 'Uneven_Bars', 'Volleyball_Spiking', 'Walking_With_Dog', 'Wall_Pushups', 'Writing_On_Board', 'Yo_Yo']

aircraft_classes = ['707-320', '727-200', '737-200', '737-300', '737-400', '737-500', '737-600', '737-700', '737-800', '737-900', '747-100', '747-200', '747-300', '747-400', '757-200', '757-300', '767-200', '767-300', '767-400', '777-200', '777-300', 'A300B4', 'A310', 'A318', 'A319', 'A320', 'A321', 'A330-200', 'A330-300', 'A340-200', 'A340-300', 'A340-500', 'A340-600', 'A380', 'ATR-42', 'ATR-72', 'An-12', 'BAE 146-200', 'BAE 146-300', 'BAE-125', 'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47', 'CRJ-200', 'CRJ-700', 'CRJ-900', 'Cessna 172', 'Cessna 208', 'Cessna 525', 'Cessna 560',
                    'Challenger 600', 'DC-10', 'DC-3', 'DC-6', 'DC-8', 'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6', 'DHC-8-100', 'DHC-8-300', 'DR-400', 'Dornier 328', 'E-170', 'E-190', 'E-195', 'EMB-120', 'ERJ 135', 'ERJ 145', 'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16A/B', 'F/A-18', 'Falcon 2000', 'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express', 'Gulfstream IV', 'Gulfstream V', 'Hawk T1', 'Il-76', 'L-1011', 'MD-11', 'MD-80', 'MD-87', 'MD-90', 'Metroliner', 'Model B200', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340', 'Spitfire', 'Tornado', 'Tu-134', 'Tu-154', 'Yak-42']

eurosat_classes = ['Annual Crop Land', 'Forest', 'Herbaceous Vegetation Land', 'Highway or Road',
                   'Industrial Buildings', 'Pasture Land', 'Permanent Crop Land', 'Residential Buildings', 'River', 'Sea or Lake']
