<!DOCTYPE html>
<html>
<head>
	<meta charset="utf8">
	<title>KnjižnicaApp</title>
	<link rel="stylesheet" href="<?php echo __SITE_URL;?>/css/style.css">
</head>
<body>
	<h1><?php echo $title; ?></h1>

	<nav>
		<ul>
			<li><a href="<?php echo __SITE_URL; ?>/users">Popis svih korisnika</a></li>
			<li><a href="<?php echo __SITE_URL; ?>/books">Popis svih knjiga</a></li>
			<li><a href="<?php echo __SITE_URL; ?>/books/search">Tražilica knjiga po autoru</a></li>
		</ul>
	</nav>
