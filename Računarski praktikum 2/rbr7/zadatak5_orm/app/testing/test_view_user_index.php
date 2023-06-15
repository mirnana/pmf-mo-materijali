<?php 

require_once __DIR__ . '/../../model/user.class.php';

$title = 'Popis korisnika';

$userList = [];
$userList[] = new User( 1, 'Pero', 'Perić', '123' );
$userList[] = new User( 2, 'Ana', 'Anić', '123' );
$userList[] = new User( 3, 'Maja', 'Majić', '123' );

require_once __DIR__ . '/../../view/users_index.php';

?>

