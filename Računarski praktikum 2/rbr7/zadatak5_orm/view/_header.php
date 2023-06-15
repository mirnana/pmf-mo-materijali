<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Library</title>

    <style>
        table { border-collapse: collapse; }
        td, th { border: solid 1px black; }
    </style>
</head>
<body>
    <h1>Knjižnica</h1>

    <hr>
        <ul>
            <li>
                <a href="index.php?rt=users/index">Popis svih korisnika</a>
            </li>
            <li>
                <a href="index.php?rt=books/index">Popis svih knjiga</a>
            </li>
            <li>
                <a href="index.php?rt=books/search">Pretraživanje knjiga po autoru</a>
            </li>
            <li>
                <a href="index.php?rt=loans/index">Popis svih posudbi</a>
            </li>
        </ul>
    <hr>

    <h2><?php echo $title; ?></h2>
    