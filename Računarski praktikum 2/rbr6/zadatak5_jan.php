<?php

renderForm();

if (!isset($_POST['ime']) || !isset($_POST['pass'])) {
    echo 'no user data';
    exit();
}

$ime = $_POST['ime'];
$pass = $_POST['pass'];

$db_user = 'student';
$db_pass = 'pass.mysql';
$db = new PDO ('mysql:host=rp2.studenti.math.hr;dbname=imrovic;charset=utf8', $db_user, $db_pass);

if(isset($_POST['ime']) && isset($_POST['pass'])){
    if (isset($_POST['login'])) {
        $q = $db->prepare('SELECT pass FROM users WHERE ime = :ime ;');
        $q->execute(array('ime' => $ime));
        if ($q->rowCount() !== 0) {
            $row = $q->fetch();
            if (password_verify($pass, $row['pass'])) {
                echo 'uspjesan login';
            }
        } else {
            echo 'ne postoji user';
        }
    } elseif (isset($_POST['novi'])) {
        $q = $db->prepare("SELECT ime, pass FROM users WHERE ime = ':ime' ;");
        $q->execute(array('ime' => $ime));
        if ($q->rowCount() === 0) {
            $hash = password_hash($pass, PASSWORD_DEFAULT);
            echo 'here 4';
            $q = $db->prepare("INSERT INTO users VALUES (':ime', ':pass' );");
            echo 'here 5';
            $q->execute(array('ime'=> $ime, 'pass'=>$hash));
            echo 'here 6';
        }
    }
}
function renderForm() {
    ?>

<!DOCTYPE html>
<html lang="hr">

<head>
</head>

<body>
    <form action="zadatak5.php" method="POST">
        Unesi ime<input type="text" name="ime"> </input>
        <br>
        Unesi pass<input type="password" name="pass"> </input>
        <br>
        <input type="submit" name="login" value="login">
        <br>
        <input type="submit" name="novi" value="Stvori novog korisnika">
    </form>

    <br />        
</body>

</html>
<?php    
}

?>

