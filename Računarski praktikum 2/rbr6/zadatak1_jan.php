<?php

upisi_trenutnog();
ispisi_zadnjih_n(5);

function upisi_trenutnog() {
    if (isset($_POST['ime'])) {
        $novi_korisnici = zapisi_korisnika($_POST['ime']);
        $f = implode(',', $novi_korisnici);
        file_put_contents('users.txt', $f);
    }
}

function ispisi_zadnjih_n($n) {
    $users = zapisani_korisnici();

    echo '<ul>';
    for ($i = 0; $i < $n; ++$i) {
        echo '<li>' . $users[$i] . '</li>';
    }
    echo '</ul>';
}

function zapisani_korisnici() {
    $f = file_get_contents('users.txt');
    return explode(',', $f);
}

function zapisi_korisnika($ime) {
    $users = zapisani_korisnici();
    array_unshift($users, $ime);
    if (isset($users[5])) {
        unset($users[5]);
    }
    return $users;
}

?>

<!DOCTYPE html>
<html lang="hr">

<head>
</head>

<body>
    Unesi svoje ime: 

    <form action="zadnji_useri.php" method="POST">
        <input type="text" name="ime"> </input>
        <br>
        <input type="submit" value="Posalji">
    </form>

    <br />        
</body>

</html>
