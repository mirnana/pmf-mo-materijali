<?php

$ucionice = [
    ['naziv' => '(003)',  'nrows' => 9, 'ncols' => 6 ],
    ['naziv' => '(006)',  'nrows' => 8, 'ncols' => 4 ],
    ['naziv' => '(001)',  'nrows' => 7, 'ncols' => 5 ],
    ['naziv' => '(101)',  'nrows' => 6, 'ncols' => 6 ],
    ['naziv' => '(A201)', 'nrows' => 5, 'ncols' => 3 ],
];

$studenti = [
    'Mirko Mirković'        => ['(006)',  3, 2],
    'Franjo Franjić'        => ['(003)',  6, 1],
    'Lucija Lucić'          => ['(A201)', 5, 2],
    'Boris Borisović'       => ['(101)',  4, 5],
    'Pero Perić'            => ['(006)',  6, 1],
    'Ana Anić'              => ['(006)',  2, 3],
    'Zrinka Zrinkić'        => ['(001)',  2, 5],
    'Slavko Slavić'         => ['(A201)', 1, 1],
    'Marko Marković'        => ['(003)',  2, 6],
    'Marija Marijić'        => ['(A201)', 2, 6],
    'Maja Majić'            => ['(006)',  8, 4],
    'Vladimir Vladimirović' => ['(101)',  3, 1],
];


// ----------------------------------
// Dodajte svoj kod ispod.
// ----------------------------------

function sendJSONandExit( $message ){
    // Kao izlaz skripte pošalji $message u JSON formatu i prekini izvođenje.
    header( 'Content-type:application/json;charset=utf-8' );
    echo json_encode( $message );
    flush();
    exit( 0 );
}

// vraćanje popisa učionica:
if(strpos($_GET['sve'], "daj ucionice")) {
    $message = [];
    foreach($ucionice as $u) echo "<option value='" . $u . "' />\n";
}

// vraćanje dimenzija tražene učionice i popisa studenata u istoj
$message = [];
foreach($ucionice as $u) {
    if(strpos($_GET['jedna'], $u['naziv'])) {
        $message['x'] =̣ $u['nrows'];
        $message['y'] =̣ $u['ncols'];
    }
}
foreach ($studenti as $s) {
    if(strpos($_GET['jedna'], $s[1][0]))
        $message['studenti'] = $s;
}
sendJSONandExit($message);


    
?>