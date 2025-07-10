import { useState } from 'react'
import './DashBoard.css'
import MainPage from './FunctionalPage/MainPage'
import WatchList from './WatchList/WatchList'

function DashBoard() {

  return (
    <>
      <MainPage />
      <WatchList />
    </>
  )
}

export default DashBoard
